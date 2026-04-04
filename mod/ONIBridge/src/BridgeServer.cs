using System;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using System.Collections.Concurrent;
using Newtonsoft.Json;
using UnityEngine;

namespace ONIBridge
{
    /// <summary>
    /// TCP server running on a background thread inside the ONI process.
    /// Accepts one AI agent connection at a time.
    ///
    /// Protocol (newline-delimited JSON):
    ///   Game → Agent: { "type": "state", "data": { ... } }  — every N game ticks
    ///   Agent → Game: { "type": "action", "action": "place_building", ... }
    ///   Game → Agent: { "type": "ack", "success": true }
    ///
    /// IMPORTANT: SendState and SendAck enqueue messages to _sendQueue.
    /// A dedicated background send thread drains the queue — the main game
    /// thread never blocks on TCP writes.
    /// </summary>
    public class BridgeServer
    {
        public static readonly BridgeServer Instance = new BridgeServer();

        private TcpListener? _listener;
        private Thread? _listenerThread;
        private Thread? _sendThread;
        private TcpClient? _connectedClient;
        private readonly ConcurrentQueue<ActionCommand> _pendingActions = new ConcurrentQueue<ActionCommand>();
        private readonly ConcurrentQueue<byte[]> _sendQueue = new ConcurrentQueue<byte[]>();
        private bool _running = false;

        // State messages are evicted (oldest dropped) when queue exceeds this depth.
        // Acks are never dropped — they are small and infrequent.
        private const int MaxStateQueueDepth = 3;

        public bool IsConnected => _connectedClient?.Connected == true;

        public void Start(int port)
        {
            if (_running) return;
            _running = true;

            _listenerThread = new Thread(() => ListenLoop(port))
            {
                IsBackground = true,
                Name = "ONIBridge-Listener"
            };
            _listenerThread.Start();

            _sendThread = new Thread(SendLoop)
            {
                IsBackground = true,
                Name = "ONIBridge-Send"
            };
            _sendThread.Start();

            Debug.Log($"[ONIBridge] Listening on port {port}");
        }

        public void Stop()
        {
            _running = false;
            _listener?.Stop();
            _connectedClient?.Close();
        }

        /// <summary>
        /// Called from the main game thread each tick to drain pending actions.
        /// </summary>
        public void DrainActions()
        {
            while (_pendingActions.TryDequeue(out var cmd))
            {
                ActionExecutor.Execute(cmd);
            }
        }

        /// <summary>
        /// Enqueue a state snapshot for async sending. Never blocks the game thread.
        /// </summary>
        public void SendState(object statePayload)
        {
            if (_connectedClient?.Connected != true) return;
            // Evict oldest state messages to keep only the freshest few.
            // The Python runner always wants the latest state — stale ones are noise.
            while (_sendQueue.Count >= MaxStateQueueDepth)
                _sendQueue.TryDequeue(out _);
            var msg = JsonConvert.SerializeObject(new { type = "state", data = statePayload });
            _sendQueue.Enqueue(Encoding.UTF8.GetBytes(msg + "\n"));
        }

        /// <summary>
        /// Enqueue an action acknowledgement for async sending. Never blocks the game thread.
        /// </summary>
        public void SendAck(string action, bool success, string? error = null)
        {
            if (_connectedClient?.Connected != true) return;
            var msg = JsonConvert.SerializeObject(new AckMessage
            {
                Action = action,
                Success = success,
                Error = error
            });
            _sendQueue.Enqueue(Encoding.UTF8.GetBytes(msg + "\n"));
        }

        /// <summary>
        /// Background thread: drains _sendQueue and writes to the TCP stream.
        /// All blocking TCP writes happen here, never on the game thread.
        /// </summary>
        private void SendLoop()
        {
            while (_running)
            {
                if (_sendQueue.TryDequeue(out var bytes))
                {
                    var client = _connectedClient;
                    if (client?.Connected != true)
                    {
                        // Client gone — discard and keep draining to prevent queue buildup
                        continue;
                    }
                    try
                    {
                        client.GetStream().Write(bytes, 0, bytes.Length);
                    }
                    catch (Exception ex)
                    {
                        Debug.LogWarning($"[ONIBridge] Send failed: {ex.Message}");
                        _connectedClient = null;
                    }
                }
                else
                {
                    Thread.Sleep(5); // nothing to send — yield briefly
                }
            }
        }

        private void ListenLoop(int port)
        {
            _listener = new TcpListener(IPAddress.Any, port);
            _listener.Start();

            while (_running)
            {
                try
                {
                    var client = _listener.AcceptTcpClient();
                    Debug.Log("[ONIBridge] AI agent connected.");
                    // Clear stale send queue from previous session
                    while (_sendQueue.TryDequeue(out _)) { }
                    // Close stale previous connection, then swap in the new one.
                    var previous = _connectedClient;
                    _connectedClient = client;
                    previous?.Close();
                    var t = new Thread(() => HandleClient(client))
                    {
                        IsBackground = true,
                        Name = "ONIBridge-Client"
                    };
                    t.Start();
                }
                catch (SocketException) when (!_running)
                {
                    break;
                }
                catch (Exception ex)
                {
                    Debug.LogWarning($"[ONIBridge] Connection error: {ex.Message}");
                }
            }
        }

        private void HandleClient(TcpClient client)
        {
            var buffer = new byte[4096];
            var stream = client.GetStream();
            var partial = new StringBuilder();

            while (_running && client.Connected)
            {
                try
                {
                    int bytesRead = stream.Read(buffer, 0, buffer.Length);
                    if (bytesRead == 0) break;

                    partial.Append(Encoding.UTF8.GetString(buffer, 0, bytesRead));
                    var raw = partial.ToString();

                    int newline;
                    while ((newline = raw.IndexOf('\n')) >= 0)
                    {
                        var line = raw.Substring(0, newline).Trim();
                        raw = raw.Substring(newline + 1);

                        if (!string.IsNullOrEmpty(line))
                            ProcessMessage(line);
                    }
                    partial.Clear();
                    partial.Append(raw);
                }
                catch (Exception ex)
                {
                    Debug.LogWarning($"[ONIBridge] Client read error: {ex.Message}");
                    break;
                }
            }

            Debug.Log("[ONIBridge] AI agent disconnected.");
            _connectedClient = null;
        }

        private void ProcessMessage(string json)
        {
            try
            {
                var cmd = JsonConvert.DeserializeObject<ActionCommand>(json);
                if (cmd != null)
                    _pendingActions.Enqueue(cmd);
            }
            catch (Exception ex)
            {
                Debug.LogWarning($"[ONIBridge] Bad message: {ex.Message} — raw: {json}");
            }
        }
    }
}
