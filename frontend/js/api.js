const API_BASE_URL = "http://127.0.0.1:8000";

const api = {
  async sendMessage(message, conversationId = null) {
    return this.request("/api/chat", {
      method: "POST",
      body: JSON.stringify({
        ...(conversationId ? { conversation_id: conversationId } : {}),
        message
      })
    });
  },
  async createConversation() {
    return this.request("/api/conversations", { method: "POST" });
  },
  async getConversations() {
    return this.request("/api/conversations");
  },
  async getConversation(conversationId) {
    return this.request(`/api/conversations/${encodeURIComponent(conversationId)}`);
  },
  async deleteConversation(conversationId) {
    return this.request(`/api/conversations/${encodeURIComponent(conversationId)}`, {
      method: "DELETE"
    });
  },
  async request(path, options = {}) {
    const response = await fetch(`${API_BASE_URL}${path}`, {
      ...options,
      headers: { "Content-Type": "application/json", ...(options.headers || {}) }
    });
    const data = await response.json().catch(() => ({}));
    if (!response.ok) {
      throw new Error(data.detail || "The assistant could not process your message.");
    }
    return data;
  }
};

window.AIAssistantAPI = api;
