document.addEventListener("DOMContentLoaded", () => {
  Chat.init();
  const sidebarToggle = document.querySelector("[data-sidebar-toggle]");
  sidebarToggle?.addEventListener("click", () => UI.setSidebar(!UI.elements.sidebar.classList.contains("is-open")));
  document.querySelectorAll("[data-sidebar-close]").forEach((button) => {
    button.addEventListener("click", () => UI.setSidebar(false));
  });
  document.getElementById("settings-button").addEventListener("click", () => {
    UI.toast("Settings will be available in a future workspace update.");
  });
  document.addEventListener("keydown", (event) => {
    if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === "k") {
      event.preventDefault();
      document.getElementById("new-chat-button").click();
    }
    if (event.key === "Escape") UI.setSidebar(false);
  });
});