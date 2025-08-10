use anyhow::Result;
use async_stream::try_stream;
use async_trait::async_trait;
use futures::TryStreamExt;
use reqwest::StatusCode;
use serde_json::Value;
use std::io;
use tokio::pin;
use tokio_util::io::StreamReader;

use super::api_client::{ApiClient, ApiResponse, AuthMethod};
use super::base::{ConfigKey, MessageStream, ModelInfo, Provider, ProviderMetadata, ProviderUsage};
use super::errors::ProviderError;
use super::formats::anthropic::{
    create_request, get_usage, response_to_message, response_to_streaming_message,
};
use super::utils::{emit_debug_trace, get_model, map_http_error_to_provider_error};
use crate::conversation::message::Message;
use crate::impl_provider_default;
use crate::model::ModelConfig;
use crate::providers::retry::ProviderRetry;
use rmcp::model::Tool;

const ANTHROPIC_DEFAULT_MODEL: &str = "claude-sonnet-4-0";
const ANTHROPIC_KNOWN_MODELS: &[&str] = &[
    "claude-sonnet-4-0",
    "claude-sonnet-4-20250514",
    "claude-opus-4-0",
    "claude-opus-4-20250514",
    "claude-3-7-sonnet-latest",
    "claude-3-7-sonnet-20250219",
    "claude-3-5-sonnet-latest",
    "claude-3-5-haiku-latest",
    "claude-3-opus-latest",
];

const ANTHROPIC_DOC_URL: &str = "https://docs.anthropic.com/en/docs/about-claude/models";
const ANTHROPIC_API_VERSION: &str = "2023-06-01";

#[derive(serde::Serialize)]
pub struct AnthropicProvider {
    #[serde(skip)]
    api_client: ApiClient,
    model: ModelConfig,
    #[serde(skip)]
    oauth_mode: bool,
}

impl_provider_default!(AnthropicProvider);

struct AnthropicOAuthAuthProvider;

#[async_trait]
impl super::api_client::AuthProvider for AnthropicOAuthAuthProvider {
    async fn get_auth_header(&self) -> anyhow::Result<(String, String)> {
        let token = get_anthropic_oauth_access_token().await?;
        Ok(("Authorization".to_string(), format!("Bearer {}", token)))
    }
}

async fn get_anthropic_oauth_access_token() -> anyhow::Result<String> {
    use std::time::{SystemTime, UNIX_EPOCH};

    const CLIENT_ID: &str = "9d1c250a-e61b-44d9-88ed-5944d1962f5e";
    const TOKEN_URL: &str = "https://console.anthropic.com/v1/oauth/token";

    let cfg = crate::config::Config::global();
    let mut access = cfg.get_secret::<String>("ANTHROPIC_OAUTH_ACCESS").unwrap_or_default();
    let refresh = cfg.get_secret::<String>("ANTHROPIC_OAUTH_REFRESH").map_err(|e| anyhow::anyhow!(e.to_string()))?;
    let expires_ms = cfg
        .get_secret::<serde_json::Number>("ANTHROPIC_OAUTH_EXPIRES")
        .ok()
        .and_then(|n| n.as_i64())
        .unwrap_or(0);
    let now_ms = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_millis() as i64;

    if !access.is_empty() && expires_ms > now_ms + 30_000 {
        return Ok(access);
    }

    // Refresh
    let client = reqwest::Client::new();
    let payload = serde_json::json!({
        "grant_type": "refresh_token",
        "refresh_token": refresh,
        "client_id": CLIENT_ID,
    });
    let resp = client
        .post(TOKEN_URL)
        .header(reqwest::header::CONTENT_TYPE, "application/json")
        .json(&payload)
        .send()
        .await?
        .error_for_status()?;
    let json: serde_json::Value = resp.json().await?;
    access = json
        .get("access_token")
        .and_then(|v| v.as_str())
        .ok_or_else(|| anyhow::anyhow!("missing access_token"))?
        .to_string();
    let new_refresh = json
        .get("refresh_token")
        .and_then(|v| v.as_str())
        .unwrap_or(&refresh)
        .to_string();
    let expires_in = json.get("expires_in").and_then(|v| v.as_i64()).unwrap_or(3600);
    let new_expires_ms = now_ms + expires_in * 1000;

    cfg.set_secret("ANTHROPIC_OAUTH_ACCESS", serde_json::Value::String(access.clone()))
        .map_err(|e| anyhow::anyhow!(e.to_string()))?;
    cfg.set_secret("ANTHROPIC_OAUTH_REFRESH", serde_json::Value::String(new_refresh))
        .map_err(|e| anyhow::anyhow!(e.to_string()))?;
    cfg.set_secret(
        "ANTHROPIC_OAUTH_EXPIRES",
        serde_json::Value::Number(new_expires_ms.into()),
    )
    .map_err(|e| anyhow::anyhow!(e.to_string()))?;

    Ok(access)
}

impl AnthropicProvider {
    pub fn from_env(model: ModelConfig) -> Result<Self> {
        let config = crate::config::Config::global();
        let host: String = config
            .get_param("ANTHROPIC_HOST")
            .unwrap_or_else(|_| "https://api.anthropic.com".to_string());

        // Detect OAuth mode by presence of refresh/access tokens
        let oauth_available = config.get_secret::<String>("ANTHROPIC_OAUTH_REFRESH").is_ok()
            || config.get_secret::<String>("ANTHROPIC_OAUTH_ACCESS").is_ok();

        let (auth, oauth_mode) = if oauth_available {
            // Use custom auth provider that injects Bearer access token
            (
                AuthMethod::Custom(Box::new(AnthropicOAuthAuthProvider {})),
                true,
            )
        } else {
            // Fallback to API key mode (required)
            let api_key: String = config.get_secret("ANTHROPIC_API_KEY")?;
            (
                AuthMethod::ApiKey {
                    header_name: "x-api-key".to_string(),
                    key: api_key,
                },
                false,
            )
        };

        let api_client =
            ApiClient::new(host, auth)?.with_header("anthropic-version", ANTHROPIC_API_VERSION)?;

        Ok(Self { api_client, model, oauth_mode })
    }

    fn get_conditional_headers(&self) -> Vec<(&str, &str)> {
        let mut headers = Vec::new();

        let is_thinking_enabled = std::env::var("CLAUDE_THINKING_ENABLED").is_ok();
        if self.model.model_name.starts_with("claude-3-7-sonnet-") {
            if is_thinking_enabled {
                headers.push(("anthropic-beta", "output-128k-2025-02-19"));
            }
            headers.push(("anthropic-beta", "token-efficient-tools-2025-02-19"));
        }
        // In OAuth mode, set required oauth beta header
        if self.oauth_mode {
            headers.push(("anthropic-beta", "oauth-2025-04-20"));
            // Present a Claude-like user agent when using Claude Code OAuth tokens
            headers.push(("User-Agent", "Claude/1.0 (goose)"));
        }

        headers
    }

    async fn post(&self, payload: &Value) -> Result<ApiResponse, ProviderError> {
        let mut request = self.api_client.request("v1/messages");

        for (key, value) in self.get_conditional_headers() {
            request = request.header(key, value)?;
        }

        Ok(request.api_post(payload).await?)
    }

    fn anthropic_api_call_result(response: ApiResponse) -> Result<Value, ProviderError> {
        match response.status {
            StatusCode::OK => response.payload.ok_or_else(|| {
                ProviderError::RequestFailed("Response body is not valid JSON".to_string())
            }),
            _ => {
                if response.status == StatusCode::BAD_REQUEST {
                    if let Some(error_msg) = response
                        .payload
                        .as_ref()
                        .and_then(|p| p.get("error"))
                        .and_then(|e| e.get("message"))
                        .and_then(|m| m.as_str())
                    {
                        let msg = error_msg.to_string();
                        if msg.to_lowercase().contains("too long")
                            || msg.to_lowercase().contains("too many")
                        {
                            return Err(ProviderError::ContextLengthExceeded(msg));
                        }
                    }
                }
                Err(map_http_error_to_provider_error(
                    response.status,
                    response.payload,
                ))
            }
        }
    }
}

#[async_trait]
impl Provider for AnthropicProvider {
    fn metadata() -> ProviderMetadata {
        let models: Vec<ModelInfo> = ANTHROPIC_KNOWN_MODELS
            .iter()
            .map(|&model_name| ModelInfo::new(model_name, 200_000))
            .collect();

        ProviderMetadata::with_models(
            "anthropic",
            "Anthropic",
            "Claude and other models from Anthropic",
            ANTHROPIC_DEFAULT_MODEL,
            models,
            ANTHROPIC_DOC_URL,
            vec![
                ConfigKey::new("ANTHROPIC_API_KEY", true, true, None),
                // Enable Anthropic OAuth (Claude Pro/Max) as an alternative auth path
                ConfigKey::new_oauth("ANTHROPIC_OAUTH", true, true, None),
                ConfigKey::new(
                    "ANTHROPIC_HOST",
                    true,
                    false,
                    Some("https://api.anthropic.com"),
                ),
            ],
        )
    }

    fn get_model_config(&self) -> ModelConfig {
        self.model.clone()
    }

    #[tracing::instrument(
        skip(self, system, messages, tools),
        fields(model_config, input, output, input_tokens, output_tokens, total_tokens)
    )]
    async fn complete(
        &self,
        system: &str,
        messages: &[Message],
        tools: &[Tool],
    ) -> Result<(Message, ProviderUsage), ProviderError> {
        // In OAuth mode, replace system prompt with a minimal Claude Code identity
        let system = if self.oauth_mode {
            "You are Claude Code, Anthropic's official CLI for Claude.".to_string()
        } else {
            system.to_string()
        };
        let payload = create_request(&self.model, &system, messages, tools)?;

        let response = self
            .with_retry(|| async { self.post(&payload).await })
            .await?;

        let json_response = Self::anthropic_api_call_result(response)?;

        let message = response_to_message(&json_response)?;
        let usage = get_usage(&json_response)?;
        tracing::debug!("🔍 Anthropic non-streaming parsed usage: input_tokens={:?}, output_tokens={:?}, total_tokens={:?}",
                usage.input_tokens, usage.output_tokens, usage.total_tokens);

        let model = get_model(&json_response);
        emit_debug_trace(&self.model, &payload, &json_response, &usage);
        let provider_usage = ProviderUsage::new(model, usage);
        tracing::debug!(
            "🔍 Anthropic non-streaming returning ProviderUsage: {:?}",
            provider_usage
        );
        Ok((message, provider_usage))
    }

    async fn configure_oauth(&self) -> Result<(), ProviderError> {
        // PKCE OAuth flow for Anthropic (Claude Pro/Max)
        use base64::{engine::general_purpose::URL_SAFE_NO_PAD as B64_URL, Engine as _};
        use sha2::{Digest, Sha256};
        use std::time::{SystemTime, UNIX_EPOCH};

        const CLIENT_ID: &str = "9d1c250a-e61b-44d9-88ed-5944d1962f5e";
        const AUTH_URL: &str = "https://claude.ai/oauth/authorize";
        const TOKEN_URL: &str = "https://console.anthropic.com/v1/oauth/token";
        const REDIRECT_URI: &str = "https://console.anthropic.com/oauth/code/callback";
        const SCOPES: &str = "org:create_api_key user:profile user:inference";

        // Generate PKCE verifier and challenge
        let verifier_bytes: [u8; 32] = rand::random();
        let verifier = B64_URL.encode(verifier_bytes);
        let challenge = {
            let mut hasher = Sha256::new();
            hasher.update(verifier.as_bytes());
            let digest = hasher.finalize();
            B64_URL.encode(digest)
        };

        // Build authorization URL
        let mut url = url::Url::parse(AUTH_URL).map_err(|e| ProviderError::ExecutionError(e.to_string()))?;
        url.query_pairs_mut()
            .append_pair("code", "true")
            .append_pair("client_id", CLIENT_ID)
            .append_pair("response_type", "code")
            .append_pair("redirect_uri", REDIRECT_URI)
            .append_pair("scope", SCOPES)
            .append_pair("code_challenge", &challenge)
            .append_pair("code_challenge_method", "S256")
            .append_pair("state", &verifier);

        // Open browser; if fails, print URL
        let open_ok = webbrowser::open(url.as_str()).is_ok();
        if !open_ok {
            println!("Open this URL to authorize Anthropic OAuth:\n{}", url);
        }
        println!("Paste the authorization code here (including any #state if present): ");
        let mut input = String::new();
        std::io::stdin().read_line(&mut input).map_err(|e| ProviderError::ExecutionError(e.to_string()))?;
        let code = input.trim();
        let mut splits = code.split('#');
        let auth_code = splits.next().unwrap_or("");
        let state = splits.next();

        // Exchange code for tokens
        let client = reqwest::Client::new();
        let payload = serde_json::json!({
            "code": auth_code,
            "state": state.unwrap_or(""),
            "grant_type": "authorization_code",
            "client_id": CLIENT_ID,
            "redirect_uri": REDIRECT_URI,
            "code_verifier": verifier,
        });
        let resp = client
            .post(TOKEN_URL)
            .header(reqwest::header::CONTENT_TYPE, "application/json")
            .json(&payload)
            .send()
            .await
            .map_err(|e| ProviderError::ExecutionError(e.to_string()))?
            .error_for_status()
            .map_err(|e| ProviderError::Authentication(e.to_string()))?;
        let json: serde_json::Value = resp
            .json()
            .await
            .map_err(|e| ProviderError::ExecutionError(e.to_string()))?;

        let access = json.get("access_token").and_then(|v| v.as_str()).ok_or_else(|| ProviderError::Authentication("missing access_token".to_string()))?;
        let refresh = json.get("refresh_token").and_then(|v| v.as_str()).ok_or_else(|| ProviderError::Authentication("missing refresh_token".to_string()))?;
        let expires_in = json.get("expires_in").and_then(|v| v.as_i64()).unwrap_or(3600);
        let now_ms = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_millis() as i64;
        let expires_ms = now_ms + expires_in * 1000;

        let cfg = crate::config::Config::global();
        cfg.set_secret("ANTHROPIC_OAUTH_ACCESS", serde_json::Value::String(access.to_string()))
            .map_err(|e| ProviderError::ExecutionError(e.to_string()))?;
        cfg.set_secret("ANTHROPIC_OAUTH_REFRESH", serde_json::Value::String(refresh.to_string()))
            .map_err(|e| ProviderError::ExecutionError(e.to_string()))?;
        cfg.set_secret("ANTHROPIC_OAUTH_EXPIRES", serde_json::Value::Number(expires_ms.into()))
            .map_err(|e| ProviderError::ExecutionError(e.to_string()))?;

        println!("Login successful");
        Ok(())
    }

    async fn fetch_supported_models(&self) -> Result<Option<Vec<String>>, ProviderError> {
        // Ensure OAuth beta header is set for model listing as well
        let mut request = self.api_client.request("v1/models");
        for (key, value) in self.get_conditional_headers() {
            request = request.header(key, value)?;
        }
        let response = request.api_get().await?;

        if response.status != StatusCode::OK {
            return Err(map_http_error_to_provider_error(
                response.status,
                response.payload,
            ));
        }

        let json = response.payload.unwrap_or_default();
        let arr = match json.get("models").and_then(|v| v.as_array()) {
            Some(arr) => arr,
            None => return Ok(None),
        };

        let mut models: Vec<String> = arr
            .iter()
            .filter_map(|m| {
                if let Some(s) = m.as_str() {
                    Some(s.to_string())
                } else if let Some(obj) = m.as_object() {
                    obj.get("id").and_then(|v| v.as_str()).map(str::to_string)
                } else {
                    None
                }
            })
            .collect();
        models.sort();
        Ok(Some(models))
    }

    async fn stream(
        &self,
        system: &str,
        messages: &[Message],
        tools: &[Tool],
    ) -> Result<MessageStream, ProviderError> {
        // In OAuth mode, replace system prompt with a minimal Claude Code identity
        let system = if self.oauth_mode {
            "You are Claude Code, Anthropic's official CLI for Claude.".to_string()
        } else {
            system.to_string()
        };
        let mut payload = create_request(&self.model, &system, messages, tools)?;
        payload
            .as_object_mut()
            .unwrap()
            .insert("stream".to_string(), Value::Bool(true));

        let mut request = self.api_client.request("v1/messages");

        for (key, value) in self.get_conditional_headers() {
            request = request.header(key, value)?;
        }

        let response = request.response_post(&payload).await?;
        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_default();
            let error_json = serde_json::from_str::<Value>(&error_text).ok();
            return Err(map_http_error_to_provider_error(status, error_json));
        }

        let stream = response.bytes_stream().map_err(io::Error::other);

        let model_config = self.model.clone();
        Ok(Box::pin(try_stream! {
            let stream_reader = StreamReader::new(stream);
            let framed = tokio_util::codec::FramedRead::new(stream_reader, tokio_util::codec::LinesCodec::new()).map_err(anyhow::Error::from);

            let message_stream = response_to_streaming_message(framed);
            pin!(message_stream);
            while let Some(message) = futures::StreamExt::next(&mut message_stream).await {
                let (message, usage) = message.map_err(|e| ProviderError::RequestFailed(format!("Stream decode error: {}", e)))?;
                emit_debug_trace(&model_config, &payload, &message, &usage.as_ref().map(|f| f.usage).unwrap_or_default());
                yield (message, usage);
            }
        }))
    }

    fn supports_streaming(&self) -> bool {
        true
    }
}
