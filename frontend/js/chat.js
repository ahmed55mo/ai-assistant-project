const Chat = {
  conversationId: null,
  processing: false,
  userId: "anonymous",
  init() {
    UI.cache();
    this.bindEvents();
    UI.updateComposer();
    this.loadDocuments();
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
    document.getElementById("attachment-button").addEventListener("click", () => document.getElementById("attachment-input").click());
    document.getElementById("attachment-input").addEventListener("change", (event) => this.uploadDocument(event.target.files[0]));
    UI.elements.refreshDocuments?.addEventListener("click", () => this.loadDocuments());
    document.getElementById("microphone-button").addEventListener("click", () => UI.toast("Microphone input is ready for a future voice integration."));
    document.getElementById("new-chat-button").addEventListener("click", () => this.newConversation());
    document.querySelectorAll(".conversation-item").forEach((item) => {
      item.addEventListener("click", () => this.selectConversation(item));
    });
  },
  async uploadDocument(file) {
    if (!file) return;
    try {
      if (!this.conversationId) {
        const conversation = await window.AIAssistantAPI.createConversation();
        this.conversationId = conversation.conversation_id;
      }
      UI.toast(`Uploading ${file.name}...`);
      const data = await window.AIAssistantAPI.uploadDocument(file, this.conversationId, this.userId);
      UI.toast(`${data.filename} is ${data.status}.`);
      await this.loadDocuments();
    } catch (error) {
      UI.toast(error.message || "The document could not be uploaded.");
    } finally {
      document.getElementById("attachment-input").value = "";
    }
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
      const data = await window.AIAssistantAPI.sendMessage(message, this.conversationId, this.userId);
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
      await this.loadDocuments();
      this.addMessage("assistant", data.response, data.sources || []);
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
  addMessage(role, text, sources = []) {
    const row = document.createElement("article");
    row.className = `message-row ${role}`;
    const avatar = role === "assistant" ? '<i class="bi bi-stars"></i>' : "AM";
    const label = role === "assistant" ? "AI Assistant" : "You";
    const sourceMarkup = role === "assistant" && sources.length
      ? this.sourcesMarkup(sources)
      : "";
    row.innerHTML = `<div class="message-avatar">${avatar}</div><div class="message-column"><div class="message-meta"><span>${label}</span><time>${this.time()}</time></div><div class="message-bubble">${this.formatText(text)}</div>${sourceMarkup}</div>`;
    UI.elements.messages.appendChild(row);
    return row;
  },
  sourcesMarkup(sources) {
    const unique = [];
    const seen = new Set();
    sources.forEach((source) => {
      if (!source || !source.document_id || !source.filename) return;
      const key = source.page_number == null
        ? source.document_id
        : `${source.document_id}:${source.page_number}`;
      if (seen.has(key)) return;
      seen.add(key);
      unique.push(source);
    });
    if (!unique.length) return "";
    const cards = unique.map((source) => {
      const filename = this.escapeHtml(source.filename);
      const page = Number.isInteger(source.page_number)
        ? `<small>Page ${source.page_number}</small>`
        : "";
      return `<div class="source-card"><i class="bi bi-file-earmark-text"></i><span><strong>${filename}</strong>${page}</span></div>`;
    }).join("");
    return `<section class="message-sources" aria-label="Sources"><div class="sources-label">Sources</div><div class="sources-list">${cards}</div></section>`;
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
    this.loadDocuments();
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
    this.loadDocuments();
    UI.toast("Conversation selected. Backend history will load here.");
    UI.setSidebar(false);
  },
  async loadDocuments() {
    const panel = UI.elements.documents;
    if (!panel) return;
    if (!this.conversationId) {
      panel.innerHTML = '<p class="documents-empty"><i class="bi bi-file-earmark-text"></i><span>No documents yet.<small>Upload a PDF, TXT, or DOCX to use it with the assistant.</small></span></p>';
      return;
    }
    panel.innerHTML = '<p class="documents-loading"><i class="bi bi-arrow-repeat"></i> Loading documents...</p>';
    UI.elements.refreshDocuments?.setAttribute("disabled", "disabled");
    try {
      const data = await window.AIAssistantAPI.listDocuments(this.conversationId, this.userId);
      const documents = Array.isArray(data.documents) ? data.documents : [];
      if (!documents.length) {
        panel.innerHTML = '<p class="documents-empty"><i class="bi bi-file-earmark-text"></i><span>No documents yet.<small>Upload a PDF, TXT, or DOCX to use it with the assistant.</small></span></p>';
        return;
      }
      panel.innerHTML = documents.map((document) => this.documentMarkup(document)).join("");
      panel.querySelectorAll("[data-delete-document]").forEach((button) => {
        button.addEventListener("click", () => this.deleteDocument(button.dataset.deleteDocument, button));
      });
    } catch (error) {
      panel.innerHTML = '<p class="documents-error"><i class="bi bi-exclamation-circle"></i><span>Documents could not be loaded.<button type="button" class="documents-retry">Try again</button></span></p>';
      panel.querySelector(".documents-retry")?.addEventListener("click", () => this.loadDocuments());
    } finally {
      UI.elements.refreshDocuments?.removeAttribute("disabled");
    }
  },
  async deleteDocument(documentId, button) {
    if (!this.conversationId || button.disabled) return;
    if (!window.confirm("Delete this document? Its knowledge will no longer be available in this conversation.")) return;
    button.disabled = true;
    try {
      await window.AIAssistantAPI.deleteDocument(documentId, this.conversationId, this.userId);
      UI.toast("Document deleted.");
      await this.loadDocuments();
    } catch (error) {
      button.disabled = false;
      UI.toast(error.message || "The document could not be deleted.");
    }
  },
  documentMarkup(document) {
    const filename = this.escapeHtml(document.filename || "Untitled document");
    const type = this.fileType(document.filename, document.content_type);
    const status = this.escapeHtml(String(document.status || "unknown"));
    const chunks = Number.isFinite(document.chunk_count) ? `${document.chunk_count} chunks` : "Processing";
    return `<article class="document-item">
      <div class="document-icon"><i class="bi bi-file-earmark-text"></i></div>
      <div class="document-copy"><strong title="${filename}">${filename}</strong><small>${type} · ${this.escapeHtml(chunks)} · ${status}</small></div>
      <button class="document-delete" type="button" data-delete-document="${this.escapeHtml(document.document_id)}" aria-label="Delete ${filename}" title="Delete document"><i class="bi bi-trash3"></i></button>
    </article>`;
  },
  fileType(filename, contentType) {
    const extension = String(filename || "").split(".").pop()?.toUpperCase();
    if (extension && extension !== filename) return extension;
    if (contentType === "application/pdf") return "PDF";
    if (contentType === "text/plain") return "TXT";
    return "Document";
  },
  escapeHtml(value) {
    return String(value).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;").replace(/'/g, "&#039;");
  }
};

window.AIAssistantChat = Chat;
