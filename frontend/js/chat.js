const Chat = {
  conversationId: null,
  processing: false,
  init() {
    UI.cache();
    this.bindEvents();
    UI.updateComposer();
  },
  bindEvents() {
    const { input, send } = UI.elements;
    input.addEventListener("input", () => UI.updateComposer());
    input.addEventListener("keydown", (event) => {
      if (event.key === "Enter" && !event.shiftKey) {
        event.preventDefault();
        this.submit();
      }
    });
    send.addEventListener("click", () => this.submit());
    document.querySelectorAll("[data-suggestion]").forEach((button) => {
      button.addEventListener("click", () => {
        input.value = button.dataset.suggestion;
        UI.updateComposer();
        input.focus();
      });
    });
    document.getElementById("attachment-button").addEventListener("click", () => UI.toast("Attachment support will be connected to your workspace soon."));
    document.getElementById("microphone-button").addEventListener("click", () => UI.toast("Microphone input is ready for a future voice integration."));
    document.getElementById("new-chat-button").addEventListener("click", () => this.newConversation());
    document.querySelectorAll(".conversation-item").forEach((item) => {
      item.addEventListener("click", () => this.selectConversation(item));
    });
  },
  async submit() {
    if (this.processing) return;
    const message = UI.elements.input.value.trim();
    if (!message) return;
    this.processing = true;
    UI.showMessages();
    this.addMessage("user", message);
    UI.elements.input.value = "";
    UI.updateComposer();
    UI.setProcessing(true);
    const typing = this.addTyping();
    UI.scrollToLatest();

    try {
      const data = await window.AIAssistantAPI.sendMessage(message, this.conversationId);
      typing.remove();
      if (
        typeof data.conversation_id !== "string" ||
        !data.conversation_id.trim() ||
        typeof data.response !== "string" ||
        !data.response.trim()
      ) {
        throw new Error("The assistant returned an empty response.");
      }
      this.conversationId = data.conversation_id;
      this.addMessage("assistant", data.response);
    } catch (error) {
      typing.remove();
      const errorMessage = error instanceof TypeError
        ? "We could not reach the assistant. Please check that the backend is running and try again."
        : error.message || "We could not process your message. Please try again.";
      this.addError(errorMessage);
    } finally {
      UI.setProcessing(false);
      this.processing = false;
      UI.elements.input.focus();
      UI.scrollToLatest();
    }
  },
  addMessage(role, text) {
    const row = document.createElement("article");
    row.className = `message-row ${role}`;
    const avatar = role === "assistant" ? '<i class="bi bi-stars"></i>' : "AM";
    const label = role === "assistant" ? "AI Assistant" : "You";
    row.innerHTML = `<div class="message-avatar">${avatar}</div><div class="message-column"><div class="message-meta"><span>${label}</span><time>${this.time()}</time></div><div class="message-bubble">${this.formatText(text)}</div></div>`;
    UI.elements.messages.appendChild(row);
    return row;
  },
  addTyping() {
    const row = document.createElement("article");
    row.className = "message-row assistant";
    row.innerHTML = '<div class="message-avatar"><i class="bi bi-stars"></i></div><div class="message-column"><div class="message-meta"><span>AI Assistant</span><span>Thinking</span></div><div class="message-bubble typing-indicator" aria-label="Assistant is typing"><span></span><span></span><span></span></div></div>';
    UI.elements.messages.appendChild(row);
    return row;
  },
  addError(text) {
    const row = document.createElement("div");
    row.className = "chat-error";
    row.setAttribute("role", "status");
    row.innerHTML = `<i class="bi bi-info-circle"></i><span>${this.formatText(text)}</span>`;
    UI.elements.messages.appendChild(row);
    return row;
  },
  formatText(text) {
    const escaped = text.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
    return escaped.replace(/\n/g, "<br>");
  },
  time() {
    return new Intl.DateTimeFormat(undefined, { hour: "numeric", minute: "2-digit" }).format(new Date());
  },
  async newConversation() {
    if (this.conversationId) {
      try {
        await window.AIAssistantAPI.deleteConversation(this.conversationId);
      } catch (error) {
        const message = error instanceof Error ? error.message : "";
        if (!message.toLowerCase().includes("not found")) {
          UI.toast("The current conversation could not be cleared.");
          return;
        }
      }
    }
    this.conversationId = null;
    UI.elements.messages.innerHTML = "";
    UI.elements.messages.classList.add("d-none");
    UI.elements.empty.classList.remove("d-none");
    UI.elements.input.value = "";
    UI.updateComposer();
    document.querySelectorAll(".conversation-item").forEach((item) => item.classList.remove("active"));
    UI.setSidebar(false);
    UI.elements.input.focus();
  },
  selectConversation(item) {
    document.querySelectorAll(".conversation-item").forEach((entry) => entry.classList.remove("active"));
    item.classList.add("active");
    this.conversationId = item.dataset.conversation;
    UI.toast("Conversation selected. Backend history will load here.");
    UI.setSidebar(false);
  }
};

window.AIAssistantChat = Chat;
