const thread = document.getElementById("thread");
const emptyState = document.getElementById("empty-state");
const composer = document.getElementById("composer");
const input = document.getElementById("message-input");
const sendButton = document.getElementById("send-button");
const sendIcon = document.getElementById("send-icon");
const statusDot = document.getElementById("status-dot");
const statusText = document.getElementById("status-text");

const lightbox = document.getElementById("lightbox");
const lightboxImg = document.getElementById("lightbox-img");
const lightboxClose = document.getElementById("lightbox-close");

const SEND_ICON = `<path d="M4 12 L20 4 L14 20 L11 13 Z" fill="currentColor"/>`;
const STOP_ICON = `<rect x="6" y="6" width="12" height="12" rx="2" fill="currentColor"/>`;

let isGenerating = false;

// ------------------------------------------------------------
// Health check
// ------------------------------------------------------------
async function checkHealth() {
    try {
        const res = await fetch("/api/health");
        if (!res.ok) throw new Error("unhealthy");
        const data = await res.json();

        if (!data.qdrant_connected) {
            statusDot.className = "status__dot degraded";
            statusText.textContent = "degraded — Qdrant unreachable";
        } else {
            statusDot.className = "status__dot online";
            // statusText.textContent = `online · VRAM ${data.vram_gb} GB`;
            statusText.textContent = "online";
        }
    } catch {
        statusDot.className = "status__dot error";
        statusText.textContent = "offline — is the server running?";
    }
}

// ------------------------------------------------------------
// Generating-state toggle (Send <-> Stop)
// ------------------------------------------------------------
function setGeneratingState(active) {
    isGenerating = active;
    input.disabled = active;
    composer.classList.toggle("generating", active);
    sendIcon.innerHTML = active ? STOP_ICON : SEND_ICON;
    sendButton.setAttribute("aria-label", active ? "Stop generating" : "Send question");
    document.querySelectorAll(".sample-pill").forEach(p => p.disabled = active);
}

async function stopGeneration() {
    try {
        await fetch("/api/chat/stop", { method: "POST" });
    } catch (e) {
        console.error("Failed to send stop signal", e);
    }
}

// ------------------------------------------------------------
// Message rendering
// ------------------------------------------------------------
function appendMessage({ role, text, citations = [], refuses = false, cancelled = false, latency = null, issues = [] }) {
    emptyState.style.display = "none";

    const stateClass = cancelled ? " cancelled" : (refuses ? " refused" : "");
    const wrapper = document.createElement("div");
    wrapper.className = `message message--${role}${stateClass}`;

    const bubble = document.createElement("div");
    bubble.className = "bubble";
    bubble.textContent = text;
    wrapper.appendChild(bubble);

    if (citations.length > 0) {
        const citationRow = document.createElement("div");
        citationRow.className = "citations";

        const headerLabel = document.createElement("div");
        headerLabel.className = "suggestion-title";
        headerLabel.style.width = "100%";
        headerLabel.style.marginTop = "1rem";
        headerLabel.textContent = "Citations / References";
        citationRow.appendChild(headerLabel);

        citations.forEach((c) => {
            const card = document.createElement("div");
            card.className = "citation-card";
            card.innerHTML = `
                <img src="${c.thumbnail}" alt="${c.doc_type} ${c.doc_id}, page ${c.page_number}" class="clickable-thumb">
                <div class="citation-card__meta">
                  <div class="doc-type">${c.doc_type}</div>
                  <div>${c.doc_id} · p${c.page_number}</div>
                </div>
            `;
            card.querySelector(".clickable-thumb").addEventListener("click", () => openLightbox(c));
            citationRow.appendChild(card);
        });
        wrapper.appendChild(citationRow);
    }

    if (issues.length > 0) {
        const note = document.createElement("div");
        note.className = "grounding-note";
        note.textContent = `⚠ ${issues.join(" ")}`;
        wrapper.appendChild(note);
    }

    if (latency !== null) {
        const latencyEl = document.createElement("div");
        latencyEl.className = "latency";
        latencyEl.textContent = `${latency}s`;
        wrapper.appendChild(latencyEl);
    }

    thread.appendChild(wrapper);
    thread.scrollTop = thread.scrollHeight;
    return wrapper;
}

function appendTyping() {
    const el = document.createElement("div");
    el.className = "message message--assistant";
    el.innerHTML = `<div class="bubble typing-indicator">Reading the corpus…</div>`;
    el.id = "typing-indicator";
    thread.appendChild(el);
    thread.scrollTop = thread.scrollHeight;
    return el;
}

// ------------------------------------------------------------
// Lightbox — instant low-res preview, progressive upgrade to full
// resolution fetched on demand from the server.
// ------------------------------------------------------------
function openLightbox(citation) {
    lightbox.setAttribute("aria-hidden", "false");
    lightboxImg.src = citation.thumbnail;
    lightboxImg.classList.add("loading");

    const fullResUrl =
        `/api/citation-image?doc_id=${encodeURIComponent(citation.doc_id)}` +
        `&doc_type=${encodeURIComponent(citation.doc_type)}` +
        `&page_number=${citation.page_number}`;

    const fullImg = new Image();
    fullImg.onload = () => {
        lightboxImg.src = fullResUrl;
        lightboxImg.classList.remove("loading");
    };
    fullImg.onerror = () => {
        // Fall back silently to the thumbnail already shown — still usable.
        lightboxImg.classList.remove("loading");
    };
    fullImg.src = fullResUrl;
}

function closeLightbox() {
    lightbox.setAttribute("aria-hidden", "true");
    lightboxImg.src = "";
}

lightboxClose.addEventListener("click", closeLightbox);
lightbox.addEventListener("click", (e) => { if (e.target === lightbox) closeLightbox(); });
document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && lightbox.getAttribute("aria-hidden") === "false") closeLightbox();
});

// ------------------------------------------------------------
// Send / receive
// ------------------------------------------------------------
async function sendMessage(message) {
    appendMessage({ role: "user", text: message });
    input.value = "";
    setGeneratingState(true);

    const typingEl = appendTyping();

    try {
        const res = await fetch("/api/chat", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ message }),
        });

        if (res.status === 503) {
            typingEl.remove();
            appendMessage({ role: "assistant", text: "Server is busy with another request. Try again shortly.", refuses: true });
            setGeneratingState(false);
            return;
        }

        if (!res.ok) {
            typingEl.remove();
            appendMessage({ role: "assistant", text: "Something went wrong generating the answer.", refuses: true });
            setGeneratingState(false);
            return;
        }

        const reader = res.body.getReader();
        const decoder = new TextDecoder("utf-8");
        let buffer = "";

        let activeBubble = null;

        while (true) {
            const { value, done } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split("\n");
            buffer = lines.pop();

            for (const line of lines) {
                const cleanLine = line.trim();
                if (!cleanLine.startsWith("data: ")) continue;

                const jsonStr = cleanLine.substring(6).trim();
                try {
                    const payload = JSON.parse(jsonStr);

                    if (payload.event === "token") {
                        // Remove loading indicator on first token batch arrival
                        if (typingEl) {
                            typingEl.remove();
                        }

                        // Create or update the streaming text bubble block
                        if (!activeBubble) {
                            emptyState.style.display = "none";
                            const wrapper = document.createElement("div");
                            wrapper.className = "message message--assistant";
                            activeBubble = document.createElement("div");
                            activeBubble.className = "bubble";
                            wrapper.appendChild(activeBubble);
                            thread.appendChild(wrapper);
                        }
                        activeBubble.textContent = payload.text;
                        thread.scrollTop = thread.scrollHeight;
                    }
                    else if (payload.event === "final" || payload.event === "cancelled") {
                        if (typingEl) typingEl.remove();

                        // Clean up temporary bubble and render the final formatted response card
                        if (activeBubble && activeBubble.parentElement) {
                            activeBubble.parentElement.remove();
                        }

                        appendMessage({
                            role: "assistant",
                            text: payload.data.answer,
                            citations: payload.data.citations,
                            refuses: payload.data.refuses,
                            cancelled: payload.data.cancelled,
                            latency: payload.data.latency_seconds,
                            issues: payload.data.grounding_issues,
                        });
                    }
                    else if (payload.event === "error") {
                        if (typingEl) typingEl.remove();
                        appendMessage({ role: "assistant", text: payload.detail, refuses: true });
                    }
                } catch (err) {
                    console.error("Failed to parse event stream frame JSON", err);
                }
            }
        }
    } catch (e) {
        if (document.getElementById("typing-indicator")) typingEl.remove();
        appendMessage({ role: "assistant", text: "Couldn't reach the server. Check it's running and try again.", refuses: true });
    } finally {
        setGeneratingState(false);
        input.focus();
    }
}

document.querySelectorAll(".sample-pill").forEach(pill => {
    pill.addEventListener("click", () => {
        if (isGenerating) return;
        const questionText = pill.textContent.trim();
        if (questionText) sendMessage(questionText);
    });
});

composer.addEventListener("submit", (e) => {
    e.preventDefault();

    if (isGenerating) {
        // Same button doubles as Stop while a request is in flight —
        // the original sendMessage() call is still awaiting the server's
        // response, which will arrive promptly once cancellation lands.
        stopGeneration();
        return;
    }

    const message = input.value.trim();
    if (!message) return;
    sendMessage(message);
});

checkHealth();
setInterval(checkHealth, 15000);