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
    /// </summary>
    public class BridgeServer
    {
        public static readonly BridgeServer Instance = new BridgeServer();

        private TcpListener? _listener;
        private Thread? _listenerThread;
        private TcpClient? _connectedClient;
        private readonly ConcurrentQueue<ActionCommand> _pendingActions = new ConcurrentQueue<ActionCommand>();
        private bool _running = false;

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
        /// Send a state snapshot to the connected AI agent.
        /// Called from the main game thread.
        /// </summary>
        public void SendState(object statePayload)
        {
            if (_connectedClient?.Connected != true) return;

            try
            {
                var msg = JsonConvert.SerializeObject(new { type = "state", data = statePayload });
                var bytes = Encoding.UTF8.GetBytes(msg + "\n");
                _connectedClient.GetStream().Write(bytes, 0, bytes.Length);
            }
            catch (Exception ex)
            {
                Debug.LogWarning($"[ONIBridge] SendState failed: {ex.Message}");
                _connectedClient = null;
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
                    _connectedClient = client;
                    HandleClient(client);
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
