//! Regression tests: a stateless HTTP backend that is briefly unreachable
//! must not leave the client permanently disconnected.
//!
//! A transient `transport.send()` failure (connection refused, DNS blip,
//! a restart) used to end the client's message loop, because
//! `HttpClientTransport::reconnect` answered `Ok(false)` when there was no
//! session to resume. Every later request then failed with
//! `Transport error: Connection closed` without touching the network.

#![cfg(all(feature = "http", feature = "http-client"))]

use std::net::SocketAddr;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use axum::Router;
use axum::http::{HeaderMap, StatusCode, header};
use axum::response::{IntoResponse, Response};
use axum::routing::post;
use reqwest::dns::{Addrs, Name, Resolve, Resolving};
use serde_json::{Value, json};
use tower_mcp::client::{HttpClientTransport, McpClient};

/// A minimal stateless MCP server: no `Mcp-Session-Id`, plain JSON replies.
///
/// Replies carry `Connection: close` so that stopping the server leaves no
/// keep-alive connection behind and the next request has to dial again.
async fn handle(body: String) -> Response {
    let request: Value = serde_json::from_str(&body).unwrap_or(Value::Null);
    let id = request.get("id").cloned();
    let method = request.get("method").and_then(Value::as_str).unwrap_or("");

    let result = match method {
        "initialize" => json!({
            "protocolVersion": request["params"]["protocolVersion"].clone(),
            "capabilities": { "tools": {} },
            "serverInfo": { "name": "stateless-test", "version": "1.0.0" },
        }),
        "ping" => json!({}),
        "tools/list" => json!({
            "tools": [{
                "name": "echo",
                "description": "Echo a message",
                "inputSchema": { "type": "object" },
            }],
        }),
        "tools/call" => json!({
            "content": [{ "type": "text", "text": "pong" }],
        }),
        // Notifications carry no id and get no JSON-RPC reply.
        _ if id.is_none() => {
            return (StatusCode::ACCEPTED, [(header::CONNECTION, "close")]).into_response();
        }
        _ => {
            let error = json!({
                "jsonrpc": "2.0",
                "id": id,
                "error": { "code": -32601, "message": "method not found" },
            });
            return json_response(error);
        }
    };

    json_response(json!({ "jsonrpc": "2.0", "id": id, "result": result }))
}

fn json_response(value: Value) -> Response {
    let mut headers = HeaderMap::new();
    headers.insert(header::CONTENT_TYPE, "application/json".parse().unwrap());
    headers.insert(header::CONNECTION, "close".parse().unwrap());
    (StatusCode::OK, headers, value.to_string()).into_response()
}

/// A server bound to a fixed address that can be stopped and started again.
struct Backend {
    addr: SocketAddr,
    task: Option<tokio::task::JoinHandle<()>>,
}

impl Backend {
    /// Reserve an ephemeral port and leave nothing listening on it.
    async fn reserve() -> Self {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        drop(listener);
        Self { addr, task: None }
    }

    fn url(&self) -> String {
        format!("http://{}/mcp", self.addr)
    }

    async fn start(&mut self) {
        assert!(self.task.is_none(), "backend already running");
        let listener = tokio::net::TcpListener::bind(self.addr).await.unwrap();
        let app = Router::new().route("/mcp", post(handle));
        self.task = Some(tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        }));
    }

    async fn stop(&mut self) {
        if let Some(task) = self.task.take() {
            task.abort();
            let _ = task.await;
        }
    }
}

impl Drop for Backend {
    fn drop(&mut self) {
        if let Some(task) = self.task.take() {
            task.abort();
        }
    }
}

fn is_dead_client_error(message: &str) -> bool {
    message.contains("Connection closed")
}

async fn call(client: &McpClient) -> Result<(), String> {
    tokio::time::timeout(Duration::from_secs(10), client.call_tool("echo", json!({})))
        .await
        .expect("call_tool timed out")
        .map(|_| ())
        .map_err(|error| error.to_string())
}

/// The case from production: the server goes away and comes back on the same
/// address, and the same client keeps working.
#[tokio::test]
async fn stateless_backend_recovers_after_restart() {
    let mut backend = Backend::reserve().await;
    backend.start().await;

    let client = McpClient::connect(HttpClientTransport::new(backend.url()))
        .await
        .unwrap();
    client.initialize("test-client", "1.0.0").await.unwrap();
    call(&client).await.expect("call before outage");

    backend.stop().await;
    let during_outage = call(&client).await;
    assert!(
        during_outage.is_err(),
        "call must fail while the server is down"
    );

    backend.start().await;
    call(&client)
        .await
        .expect("call after the server came back must succeed on the same client");
    client.shutdown().await.unwrap();
}

/// Repeated outages must each be survivable, not only the first.
#[tokio::test]
async fn stateless_backend_survives_repeated_outages() {
    let mut backend = Backend::reserve().await;
    backend.start().await;

    let client = McpClient::connect(HttpClientTransport::new(backend.url()))
        .await
        .unwrap();
    client.initialize("test-client", "1.0.0").await.unwrap();

    for round in 0..3 {
        backend.stop().await;
        for _ in 0..2 {
            let error = call(&client).await.expect_err("server is down");
            // The failure must be a real connect error, not the poisoned
            // client's "Connection closed" that never reaches the network.
            assert!(
                !is_dead_client_error(&error),
                "round {round}: client is dead instead of reporting the outage: {error}"
            );
        }
        backend.start().await;
        call(&client)
            .await
            .unwrap_or_else(|error| panic!("round {round}: no recovery: {error}"));
    }
    client.shutdown().await.unwrap();
}

/// Connection refused on the very first request (here `initialize`) must not
/// poison the client either.
#[tokio::test]
async fn refused_connection_on_first_request_is_recoverable() {
    let mut backend = Backend::reserve().await;

    let client = McpClient::connect(HttpClientTransport::new(backend.url()))
        .await
        .unwrap();
    let first = tokio::time::timeout(
        Duration::from_secs(10),
        client.initialize("test-client", "1.0.0"),
    )
    .await
    .expect("initialize timed out");
    assert!(first.is_err(), "nothing is listening yet");

    backend.start().await;
    client
        .initialize("test-client", "1.0.0")
        .await
        .expect("initialize must succeed on the same client once the server is up");
    call(&client).await.expect("tool call after recovery");
    client.shutdown().await.unwrap();
}

/// A send error for one request must not fail the others that are fine, nor
/// surface as "Connection closed" afterwards.
#[tokio::test]
async fn failed_send_reports_the_transport_error_not_a_closed_connection() {
    let mut backend = Backend::reserve().await;
    backend.start().await;

    let client = McpClient::connect(HttpClientTransport::new(backend.url()))
        .await
        .unwrap();
    client.initialize("test-client", "1.0.0").await.unwrap();

    backend.stop().await;
    let error = call(&client).await.expect_err("server is down");
    assert!(
        !is_dead_client_error(&error),
        "first failure must carry the transport error: {error}"
    );
    assert!(client.is_connected(), "client must stay connected");
}

/// Resolver that can be told to fail, like Docker DNS during a redeploy.
struct FlakyDns {
    failing: Arc<AtomicBool>,
    addr: SocketAddr,
}

impl Resolve for FlakyDns {
    fn resolve(&self, _name: Name) -> Resolving {
        let failing = self.failing.load(Ordering::SeqCst);
        let addr = self.addr;
        Box::pin(async move {
            if failing {
                Err("dns error: failed to lookup address information".into())
            } else {
                Ok(Box::new(std::iter::once(addr)) as Addrs)
            }
        })
    }
}

/// The exact failure from the production log: the name briefly does not
/// resolve while the server itself is fine.
#[tokio::test]
async fn dns_failure_is_recoverable() {
    let mut backend = Backend::reserve().await;
    backend.start().await;

    let failing = Arc::new(AtomicBool::new(false));
    let http = reqwest::Client::builder()
        .dns_resolver(Arc::new(FlakyDns {
            failing: failing.clone(),
            addr: backend.addr,
        }))
        .build()
        .unwrap();
    let url = format!("http://hive-api.test:{}/mcp", backend.addr.port());
    let client = McpClient::connect(HttpClientTransport::with_client(url, http))
        .await
        .unwrap();
    client.initialize("test-client", "1.0.0").await.unwrap();
    call(&client).await.expect("call before the DNS failure");

    failing.store(true, Ordering::SeqCst);
    for _ in 0..2 {
        let error = call(&client).await.expect_err("name does not resolve");
        assert!(!is_dead_client_error(&error), "client is dead: {error}");
    }

    failing.store(false, Ordering::SeqCst);
    call(&client)
        .await
        .expect("call after DNS recovered must succeed on the same client");
    client.shutdown().await.unwrap();
}
