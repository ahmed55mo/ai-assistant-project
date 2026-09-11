const UI = {
  elements: {},
  cache() {
    this.elements = {
      sidebar: document.getElementById("app-sidebar"),
      backdrop: document.querySelector(".sidebar-backdrop"),
      toggle: document.querySelector("[data-sidebar-toggle]"),
      input: document.getElementById("message-input"),
      send: document.getElementById("send-button"),
      count: document.getElementById("character-count"),
      composer: document.getElementById("composer"),
      empty: document.getElementById("empty-state"),
      messages: document.getElementById("messages-list"),
      scroll: document.getElementById("messages-scroll"),
      toast: document.getElementById("toast-container")
    };
  },
  setSidebar(open) {
    this.elements.sidebar.classList.toggle("is-open", open);
    this.elements.backdrop.classList.toggle("is-visible", open);
    this.elements.toggle?.setAttribute("aria-expanded", String(open));
  },
  resizeInput() {
    const input = this.elements.input;
    input.style.height = "auto";
    input.style.height = `${Math.min(input.scrollHeight, 150)}px`;
  },
  updateComposer() {
    const length = this.elements.input.value.length;
    this.elements.count.textContent = `${length.toLocaleString()} / 4000`;
    this.elements.send.disabled = !this.elements.input.value.trim();
    this.resizeInput();
  },
  setProcessing(processing) {
    this.elements.send.disabled = processing || !this.elements.input.value.trim();
    this.elements.send.classList.toggle("is-processing", processing);
    this.elements.send.setAttribute("aria-busy", String(processing));
    this.elements.input.disabled = processing;
    document.getElementById("attachment-button").disabled = processing;
  },
  showMessages() {
    this.elements.empty.classList.add("d-none");
    this.elements.messages.classList.remove("d-none");
  },
  scrollToLatest() {
    this.elements.scroll.scrollTo({ top: this.elements.scroll.scrollHeight, behavior: "smooth" });
  },
  toast(message) {
    const node = document.createElement("div");
    node.className = "toast show toast-message";
    node.setAttribute("role", "status");
    node.innerHTML = `<div class="toast-body">${message}</div>`;
    this.elements.toast.appendChild(node);
    window.setTimeout(() => node.remove(), 2600);
  }
};

window.AIAssistantUI = UI;
