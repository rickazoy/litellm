use std::time::Duration;

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use super::transformation::{ChatCompletionsAuth, ChatCompletionsProviderConfig};

/// A `/chat/completions` call as it crosses into the core.
///
/// `optional_params` arrives already mapped to the provider's own parameter
/// names by the host, exactly as the messages route receives an already
/// Anthropic-shaped body. The core owns the conversation translation, the
/// provider call, and the response normalization.
pub struct ChatCompletionsRequest<'a> {
    pub model: &'a str,
    pub messages: Value,
    pub optional_params: Map<String, Value>,
    pub api_key: Option<&'a str>,
    pub api_base: Option<&'a str>,
    pub custom_llm_provider: Option<&'a str>,
    pub extra_headers: Option<Map<String, Value>>,
    pub timeout: Option<Duration>,
}

pub(super) struct ProviderChatCompletionsRequest {
    pub(super) model: String,
    pub(super) config: &'static dyn ChatCompletionsProviderConfig,
    pub(super) url: String,
    pub(super) body: Value,
    pub(super) upstream_headers: Vec<(String, String)>,
    pub(super) auth: ChatCompletionsAuth,
    #[cfg_attr(not(feature = "bedrock-auth"), allow(dead_code))]
    pub(super) optional_params: Map<String, Value>,
    pub(super) timeout: Option<Duration>,
}

/// The provider-shaped request body a config produces. Named rather than a bare
/// `Value` so the transform contract stays a typed one, mirroring
/// [`crate::audio_transcription::types::AudioTranscriptionRequestData`].
pub struct ProviderChatRequestData {
    pub body: Value,
}

/// The raw provider response body handed back to a config for normalization.
pub struct ProviderChatResponseData {
    pub body: Value,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ChatMessageContent {
    Text(String),
    Parts(Vec<Value>),
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub content: Option<ChatMessageContent>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

/// OpenAI `usage`, including the `prompt_tokens_details` split LiteLLM's Python
/// path reports so cost tracking sees the same numbers on either path.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct PromptTokensDetails {
    pub cached_tokens: u64,
    pub cache_creation_tokens: u64,
    pub text_tokens: u64,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ChatCompletionsUsage {
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
    pub total_tokens: u64,
    pub prompt_tokens_details: PromptTokensDetails,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ChatCompletionsChoiceMessage {
    pub role: String,
    // Python emits `content: None` rather than an empty string when a turn
    // produced no text; serialize it even when None for parity.
    pub content: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ChatCompletionsChoice {
    pub index: u64,
    pub message: ChatCompletionsChoiceMessage,
    pub finish_reason: String,
}

/// The normalized response handed back to the host.
///
/// There is deliberately no `id`: Python mints the `chatcmpl-…` id on the
/// `ModelResponse` it already created, and echoing the provider's own id here
/// would change it. Pinned by `response_carries_no_id` in `tests.rs`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ChatCompletionsResponse {
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChatCompletionsChoice>,
    pub usage: ChatCompletionsUsage,
}
