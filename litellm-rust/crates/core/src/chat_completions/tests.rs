use serde_json::{Map, Value, json};

use crate::error::CoreError;

use super::prepare::prepare_chat_completions_call;
use super::transformation::ChatCompletionsAuth;
use super::types::ChatCompletionsRequest;

fn request<'a>(
    model: &'a str,
    provider: Option<&'a str>,
    messages: Value,
    optional_params: Value,
) -> ChatCompletionsRequest<'a> {
    ChatCompletionsRequest {
        model,
        messages,
        optional_params: match optional_params {
            Value::Object(map) => map,
            other => panic!("params must be an object, got {other}"),
        },
        api_key: Some("sk-test"),
        api_base: None,
        custom_llm_provider: provider,
        extra_headers: None,
        timeout: None,
    }
}

/// `ProviderChatCompletionsRequest` deliberately has no `Debug` (its headers
/// carry resolved credentials), so unwrap the failure case by hand.
fn decline(request: ChatCompletionsRequest<'_>) -> CoreError {
    match prepare_chat_completions_call(request) {
        Err(error) => error,
        Ok(prepared) => panic!("expected a decline, prepared a call to {}", prepared.url),
    }
}

#[test]
fn resolves_the_provider_from_the_model_prefix() {
    let prepared = prepare_chat_completions_call(request(
        "anthropic/claude-sonnet-4-5",
        None,
        json!([{"role": "user", "content": "hi"}]),
        json!({"max_tokens": 16}),
    ))
    .expect("prepares");
    assert_eq!(prepared.model, "claude-sonnet-4-5");
    assert_eq!(prepared.url, "https://api.anthropic.com/v1/messages");
    assert_eq!(prepared.body["model"], json!("claude-sonnet-4-5"));
}

#[test]
fn strips_an_explicit_provider_prefix_from_the_model() {
    let prepared = prepare_chat_completions_call(request(
        "anthropic/claude-sonnet-4-5",
        Some("anthropic"),
        json!([{"role": "user", "content": "hi"}]),
        json!({}),
    ))
    .expect("prepares");
    assert_eq!(prepared.model, "claude-sonnet-4-5");
}

#[test]
fn adds_the_auth_and_default_headers() {
    let prepared = prepare_chat_completions_call(request(
        "claude-sonnet-4-5",
        Some("anthropic"),
        json!([{"role": "user", "content": "hi"}]),
        json!({}),
    ))
    .expect("prepares");
    assert!(
        prepared
            .upstream_headers
            .contains(&("x-api-key".to_string(), "sk-test".to_string()))
    );
    assert!(
        prepared
            .upstream_headers
            .contains(&("anthropic-version".to_string(), "2023-06-01".to_string()))
    );
    assert!(matches!(
        prepared.auth,
        ChatCompletionsAuth::Header {
            name: "x-api-key",
            ..
        }
    ));
}

#[test]
fn a_caller_supplied_auth_header_wins_over_the_resolved_key() {
    let mut call = request(
        "claude-sonnet-4-5",
        Some("anthropic"),
        json!([{"role": "user", "content": "hi"}]),
        json!({}),
    );
    call.extra_headers = Some(Map::from_iter([(
        "X-Api-Key".to_string(),
        json!("sk-caller"),
    )]));
    let prepared = prepare_chat_completions_call(call).expect("prepares");
    let keys: Vec<_> = prepared
        .upstream_headers
        .iter()
        .filter(|(name, _)| name.eq_ignore_ascii_case("x-api-key"))
        .collect();
    assert_eq!(keys.len(), 1, "got {:?}", prepared.upstream_headers);
    assert_eq!(keys[0].1, "sk-caller");
}

#[test]
fn declines_an_unsupported_request_before_resolving_credentials() {
    let mut call = request(
        "claude-sonnet-4-5",
        Some("anthropic"),
        json!([{"role": "user", "content": "hi"}]),
        json!({"stream": true}),
    );
    call.api_key = None;
    // No api_key is set and no env is consulted: the gate must run first, so the
    // error is the decline rather than a missing-credential error.
    assert_eq!(decline(call), CoreError::Unsupported("streaming"));
}

#[test]
fn rejects_an_unknown_provider() {
    assert_eq!(
        decline(request(
            "openai/gpt-4o",
            None,
            json!([{"role": "user", "content": "hi"}]),
            json!({}),
        )),
        CoreError::InvalidProvider("openai".to_string())
    );
}

#[test]
fn rejects_a_model_with_no_resolvable_provider() {
    assert!(matches!(
        decline(request(
            "claude-sonnet-4-5",
            None,
            json!([{"role": "user", "content": "hi"}]),
            json!({}),
        )),
        CoreError::InvalidProvider(_)
    ));
}

#[test]
fn rejects_an_empty_or_malformed_message_list() {
    assert_eq!(
        decline(request(
            "anthropic/claude-sonnet-4-5",
            None,
            json!([]),
            json!({}),
        )),
        CoreError::InvalidRequest("chat completions requires at least one message".to_string())
    );
    assert!(matches!(
        decline(request(
            "anthropic/claude-sonnet-4-5",
            None,
            json!("not a list"),
            json!({}),
        )),
        CoreError::InvalidRequest(_)
    ));
}

#[test]
fn rejects_non_string_extra_headers() {
    let mut call = request(
        "anthropic/claude-sonnet-4-5",
        None,
        json!([{"role": "user", "content": "hi"}]),
        json!({}),
    );
    call.extra_headers = Some(Map::from_iter([("x-trace".to_string(), json!(7))]));
    assert_eq!(
        decline(call),
        CoreError::InvalidRequest(
            "chat completions extra_headers.x-trace must be a string, got number".to_string()
        )
    );
}

#[cfg(feature = "bedrock-auth")]
#[test]
fn prepares_a_bedrock_call_without_resolving_credentials() {
    let prepared = prepare_chat_completions_call(request(
        "bedrock/us-east-1/anthropic.claude-v2",
        None,
        json!([{"role": "user", "content": "hi"}]),
        json!({"maxTokens": 16}),
    ))
    .expect("prepares");
    assert_eq!(
        prepared.url,
        "https://bedrock-runtime.us-east-1.amazonaws.com/model/anthropic.claude-v2/converse"
    );
    assert_eq!(
        prepared.auth,
        ChatCompletionsAuth::AwsSigV4 {
            region: "us-east-1".to_string()
        }
    );
    // SigV4 signs the serialized body, so prepare must not have added an
    // Authorization header; the handler does it.
    assert!(
        !prepared
            .upstream_headers
            .iter()
            .any(|(name, _)| name.eq_ignore_ascii_case("authorization"))
    );
    assert_eq!(prepared.body["inferenceConfig"], json!({"maxTokens": 16}));
}

fn decline_reason(
    model: &str,
    provider: Option<&str>,
    messages: Value,
    params: Value,
) -> Option<&'static str> {
    let params = match params {
        Value::Object(map) => map,
        other => panic!("params must be an object, got {other}"),
    };
    super::chat_completions_decline_reason(model, provider, messages, &params)
}

#[test]
fn the_gate_accepts_what_prepare_accepts() {
    assert_eq!(
        decline_reason(
            "anthropic/claude-sonnet-4-5",
            None,
            json!([{"role": "user", "content": "hi"}]),
            json!({"max_tokens": 16}),
        ),
        None
    );
}

#[test]
fn the_gate_declines_without_resolving_credentials_or_calling_out() {
    assert_eq!(
        decline_reason(
            "anthropic/claude-sonnet-4-5",
            None,
            json!([{"role": "user", "content": "hi"}]),
            json!({"stream": true}),
        ),
        Some("streaming")
    );
    assert_eq!(
        decline_reason(
            "openai/gpt-4o",
            None,
            json!([{"role": "user", "content": "hi"}]),
            json!({}),
        ),
        Some("provider is not on the rust chat completions path")
    );
    assert_eq!(
        decline_reason(
            "claude-sonnet-4-5",
            None,
            json!([{"role": "user", "content": "hi"}]),
            json!({}),
        ),
        Some("provider is not on the rust chat completions path")
    );
    assert_eq!(
        decline_reason(
            "anthropic/claude-sonnet-4-5",
            None,
            json!("nope"),
            json!({})
        ),
        Some("unreadable message list")
    );
    assert_eq!(
        decline_reason("anthropic/claude-sonnet-4-5", None, json!([]), json!({})),
        Some("empty message list")
    );
}

#[test]
fn the_gate_agrees_with_prepare_on_every_case_it_accepts() {
    // A gate that accepts what prepare then declines would make the host emit
    // its pre-call logging on a path that falls back, so pin the agreement.
    for (messages, params) in [
        (
            json!([{"role": "user", "content": "hi"}]),
            json!({"max_tokens": 8}),
        ),
        (
            json!([{"role": "system", "content": "s"}, {"role": "user", "content": "hi"}]),
            json!({"temperature": 0.1}),
        ),
        (
            json!([{"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"}]),
            json!({}),
        ),
    ] {
        assert_eq!(
            decline_reason(
                "anthropic/claude-sonnet-4-5",
                None,
                messages.clone(),
                params.clone()
            ),
            None,
            "gate declined {messages}"
        );
        prepare_chat_completions_call(request(
            "anthropic/claude-sonnet-4-5",
            None,
            messages.clone(),
            params,
        ))
        .unwrap_or_else(|error| panic!("prepare declined {messages}: {error}"));
    }
}

mod round_trip {
    use super::*;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::{TcpListener, TcpStream};

    use crate::chat_completions::chat_completions;

    async fn read_http_request(socket: &mut TcpStream) -> String {
        let mut request = Vec::new();
        let mut buffer = [0_u8; 1024];
        let header_end = loop {
            let n = socket.read(&mut buffer).await.expect("reads request");
            if n == 0 {
                break request.len();
            }
            request.extend_from_slice(&buffer[..n]);
            if let Some(position) = request.windows(4).position(|window| window == b"\r\n\r\n") {
                break position + 4;
            }
        };
        let headers = String::from_utf8_lossy(&request[..header_end]);
        let content_length = headers
            .lines()
            .find_map(|line| {
                let (name, value) = line.split_once(':')?;
                name.eq_ignore_ascii_case("content-length")
                    .then(|| value.trim().parse::<usize>().ok())
                    .flatten()
            })
            .unwrap_or(0);
        while request.len().saturating_sub(header_end) < content_length {
            let n = socket.read(&mut buffer).await.expect("reads body");
            if n == 0 {
                break;
            }
            request.extend_from_slice(&buffer[..n]);
        }
        String::from_utf8(request).expect("request is utf8")
    }

    fn http_response(status: &str, body: &str) -> String {
        format!(
            "HTTP/1.1 {status}\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{}",
            body.len(),
            body
        )
    }

    /// Serve one request from a stub upstream and hand back what it received.
    async fn serve_once(
        status: &'static str,
        body: &'static str,
    ) -> (String, tokio::task::JoinHandle<String>) {
        let listener = TcpListener::bind("127.0.0.1:0").await.expect("binds");
        let port = listener.local_addr().expect("addr").port();
        let handle = tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.expect("accepts");
            let received = read_http_request(&mut socket).await;
            socket
                .write_all(http_response(status, body).as_bytes())
                .await
                .expect("writes response");
            socket.flush().await.expect("flushes");
            received
        });
        (format!("http://127.0.0.1:{port}/v1/messages"), handle)
    }

    fn call(api_base: &str, messages: Value, params: Value) -> ChatCompletionsRequest<'_> {
        ChatCompletionsRequest {
            model: "anthropic/claude-sonnet-4-5",
            messages,
            optional_params: match params {
                Value::Object(map) => map,
                other => panic!("params must be an object, got {other}"),
            },
            api_key: Some("sk-test"),
            api_base: Some(api_base),
            custom_llm_provider: None,
            extra_headers: None,
            timeout: Some(std::time::Duration::from_secs(10)),
        }
    }

    const GOOD_BODY: &str = r#"{"id":"msg_1","type":"message","role":"assistant","model":"claude-sonnet-4-5-20260101","content":[{"type":"text","text":"hello"}],"stop_reason":"end_turn","stop_sequence":null,"usage":{"input_tokens":11,"output_tokens":4}}"#;

    #[tokio::test]
    async fn round_trip_sends_the_translated_body_and_normalizes_the_response() {
        let (api_base, handle) = serve_once("200 OK", GOOD_BODY).await;
        let response = chat_completions(call(
            &api_base,
            json!([
                {"role": "system", "content": "be terse"},
                {"role": "user", "content": "hi"}
            ]),
            json!({"max_tokens": 16}),
        ))
        .await
        .expect("call succeeds");

        let received = handle.await.expect("server task");
        let sent: Value = serde_json::from_str(
            received
                .split_once("\r\n\r\n")
                .expect("request has a body")
                .1,
        )
        .expect("body is json");
        assert_eq!(
            sent["messages"],
            json!([{"role": "user", "content": [{"type": "text", "text": "hi"}]}])
        );
        assert_eq!(
            sent["system"],
            json!([{"type": "text", "text": "be terse"}])
        );
        assert_eq!(sent["max_tokens"], json!(16));
        assert!(received.to_lowercase().contains("x-api-key: sk-test"));

        assert_eq!(
            response.choices[0].message.content.as_deref(),
            Some("hello")
        );
        assert_eq!(response.usage.total_tokens, 15);
    }

    #[tokio::test]
    async fn a_response_it_cannot_normalize_is_reported_as_already_sent() {
        // The provider was called and billed, so the host must not retry this
        // on its own path. `MissingField` here would read as a pre-send
        // decline and be retried; `InvalidResponse` cannot.
        const NO_USAGE: &str =
            r#"{"model":"m","content":[{"type":"text","text":"hi"}],"stop_reason":"end_turn"}"#;
        let (api_base, handle) = serve_once("200 OK", NO_USAGE).await;
        let err = chat_completions(call(
            &api_base,
            json!([{"role": "user", "content": "hi"}]),
            json!({"max_tokens": 16}),
        ))
        .await
        .expect_err("response cannot be normalized");
        handle.await.expect("server task");
        assert!(
            matches!(err, CoreError::InvalidResponse(_)),
            "expected a post-send error, got {err:?}"
        );
    }

    #[tokio::test]
    async fn a_tool_use_block_in_the_response_is_also_reported_as_already_sent() {
        const TOOL_USE: &str = r#"{"model":"m","content":[{"type":"tool_use","id":"t","name":"f","input":{}}],"stop_reason":"tool_use","usage":{"input_tokens":1,"output_tokens":1}}"#;
        let (api_base, handle) = serve_once("200 OK", TOOL_USE).await;
        let err = chat_completions(call(
            &api_base,
            json!([{"role": "user", "content": "hi"}]),
            json!({"max_tokens": 16}),
        ))
        .await
        .expect_err("response cannot be normalized");
        handle.await.expect("server task");
        assert!(
            matches!(err, CoreError::InvalidResponse(_)),
            "expected a post-send error, got {err:?}"
        );
    }

    #[tokio::test]
    async fn an_upstream_error_status_keeps_its_code() {
        let (api_base, handle) =
            serve_once("429 Too Many Requests", r#"{"error":"slow down"}"#).await;
        let err = chat_completions(call(
            &api_base,
            json!([{"role": "user", "content": "hi"}]),
            json!({"max_tokens": 16}),
        ))
        .await
        .expect_err("upstream rejects");
        handle.await.expect("server task");
        assert!(
            matches!(err, CoreError::Http { status: 429, .. }),
            "expected a 429, got {err:?}"
        );
    }

    #[test]
    fn response_errors_collapse_to_one_variant_that_can_only_mean_already_sent() {
        use crate::chat_completions::handler::as_response_error;

        for original in [
            CoreError::MissingField("usage"),
            CoreError::Unsupported("non-text response content block"),
            CoreError::InvalidRequest("whatever".to_string()),
            CoreError::Auth("whatever".to_string()),
        ] {
            let label = format!("{original:?}");
            assert!(
                matches!(as_response_error(original), CoreError::InvalidResponse(_)),
                "{label} must not stay retryable once the provider has answered"
            );
        }
        // An upstream status is already unambiguous, so it survives intact.
        assert!(matches!(
            as_response_error(CoreError::Http {
                status: 500,
                body: "boom".to_string()
            }),
            CoreError::Http { status: 500, .. }
        ));
    }
}
