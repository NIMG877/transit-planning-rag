const form = document.getElementById("chat-form");
const questionInput = document.getElementById("question");
const modeSelect = document.getElementById("mode");
const topKInput = document.getElementById("top-k");
const topKOutput = document.getElementById("top-k-output");
const submitButton = document.getElementById("submit-btn");
const settingsBtn = document.getElementById("settings-btn");
const settingsPopover = document.getElementById("settings-popover");
const detailsPanel = document.getElementById("details-panel");
const detailsToggle = document.getElementById("details-toggle");
const workspace = document.getElementById("workspace");
const chatHistory = document.getElementById("chat-history");
const chatEmpty = document.getElementById("chat-empty");
const entitiesList = document.getElementById("entities-list");
const pathsList = document.getElementById("paths-list");
const docsList = document.getElementById("docs-list");
const entitiesBlock = document.getElementById("entities-block");
const pathsBlock = document.getElementById("paths-block");
const docsBlock = document.getElementById("docs-block");
const badgeMode = document.getElementById("badge-mode");
const badgeCategory = document.getElementById("badge-category");
const quickQuestionBox = document.getElementById("quick-questions");

const quickQuestions = [
  "十四五期间上海综合交通发展的总体思路是什么？",
  "上海在轨道交通网络建设方面提出了哪些重点行动？",
  "什么是“多网融合”，在政策里如何定义？",
  "相比上一轮规划，本轮政策的重点变化有哪些？",
];

const modeText = {
  vector: "Vector RAG",
  pathrag: "PathRAG",
  hybrid: "Hybrid Multi-Recall",
};

const messages = [];
let isSettingsOpen = false;
let messageSeed = 0;

function setLoading(loading) {
  submitButton.disabled = loading;
  submitButton.classList.toggle("is-loading", loading);
  submitButton.setAttribute("aria-busy", String(loading));
  modeSelect.disabled = loading;
  topKInput.disabled = loading;
}

function clearList(target) {
  target.innerHTML = "";
}

function formatTime(timestamp) {
  return new Intl.DateTimeFormat("zh-CN", {
    hour: "2-digit",
    minute: "2-digit",
  }).format(timestamp);
}

function formatElapsed(elapsedMs) {
  return typeof elapsedMs === "number" ? `${elapsedMs} ms` : "";
}

function scrollChatToBottom() {
  chatHistory.scrollTop = chatHistory.scrollHeight;
}

function createMessage(role, content, options = {}) {
  const message = {
    id: `${role}-${Date.now()}-${messageSeed += 1}`,
    role,
    content: content || "",
    status: options.status || "done",
    data: options.data || null,
    createdAt: options.createdAt || Date.now(),
  };
  messages.push(message);
  return message;
}

function updateMessageStatus(messageId, patch) {
  const message = messages.find((item) => item.id === messageId);
  if (!message) {
    return null;
  }
  Object.assign(message, patch);
  return message;
}

function renderMessage(message) {
  const wrapper = document.createElement("section");
  wrapper.className = `chat-message ${message.role}`;
  wrapper.dataset.messageId = message.id;

  const role = document.createElement("div");
  role.className = "message-role";
  role.textContent = message.role === "user" ? "用户" : "模型";

  const bubble = document.createElement("div");
  bubble.className = "message-bubble";
  bubble.dataset.role = "bubble";
  if (message.role === "assistant") {
    bubble.classList.add("assistant-bubble");
  }
  if (message.status === "loading") {
    bubble.classList.add("loading");
  }
  bubble.textContent = message.content;

  if (message.role === "assistant" && message.status !== "loading" && message.data) {
    bubble.title = "点击查看右侧详情";
    bubble.addEventListener("click", () => {
      selectMessage(message.id);
    });
  }

  const time = document.createElement("div");
  time.className = "message-time";
  const baseTime = formatTime(message.createdAt);
  const elapsed = message.role === "assistant" ? formatElapsed(message.data && message.data.elapsed_ms) : "";
  time.textContent = elapsed ? `${baseTime} · ${elapsed}` : baseTime;

  wrapper.appendChild(role);
  wrapper.appendChild(bubble);
  wrapper.appendChild(time);
  return wrapper;
}

function renderMessages() {
  chatHistory.innerHTML = "";
  if (messages.length === 0) {
    chatEmpty.classList.remove("hidden");
    chatHistory.appendChild(chatEmpty);
    return;
  }

  chatEmpty.classList.add("hidden");
  messages.forEach((message) => {
    chatHistory.appendChild(renderMessage(message));
  });
  scrollChatToBottom();
}

function updateRenderedMessageContent(messageId, content, isLoading) {
  const wrapper = chatHistory.querySelector(`[data-message-id="${messageId}"]`);
  if (!wrapper) {
    renderMessages();
    return;
  }

  const bubble = wrapper.querySelector('[data-role="bubble"]');
  if (!bubble) {
    renderMessages();
    return;
  }

  bubble.textContent = content;
  bubble.classList.toggle("loading", Boolean(isLoading));
  scrollChatToBottom();
}

function toggleBlock(element, visible) {
  element.classList.toggle("hidden", !visible);
}

function setSettingsOpen(open) {
  isSettingsOpen = open;
  settingsPopover.classList.toggle("hidden", !open);
  settingsBtn.setAttribute("aria-expanded", String(open));
}

function setDetailsOpen(open) {
  detailsPanel.classList.toggle("hidden", !open);
  workspace.classList.toggle("details-collapsed", !open);
}

function closeSettingsOnOutsideClick(event) {
  if (!isSettingsOpen) {
    return;
  }
  if (settingsPopover.contains(event.target) || settingsBtn.contains(event.target)) {
    return;
  }
  setSettingsOpen(false);
}

function appendEntityChips(entities) {
  clearList(entitiesList);
  const hasEntities = Array.isArray(entities) && entities.length > 0;
  toggleBlock(entitiesBlock, hasEntities);
  if (!hasEntities) {
    return;
  }

  entities.forEach((entity) => {
    const item = document.createElement("li");
    item.className = "chip";
    item.textContent = entity;
    entitiesList.appendChild(item);
  });
}

function appendPathEvidence(paths) {
  clearList(pathsList);
  const hasPaths = Array.isArray(paths) && paths.length > 0;
  toggleBlock(pathsBlock, hasPaths);
  if (!hasPaths) {
    return;
  }

  paths.forEach((pathItem, index) => {
    const item = document.createElement("li");
    item.className = "trace-item";

    const main = document.createElement("p");
    main.className = "trace-main";
    main.textContent = `#${index + 1} ${pathItem.path || "(空路径)"}`;

    const meta = document.createElement("p");
    meta.className = "trace-meta";
    const scoreText = typeof pathItem.hybrid_score === "number"
      ? `hybrid=${pathItem.hybrid_score.toFixed(4)}`
      : typeof pathItem.score === "number"
        ? `score=${pathItem.score.toFixed(4)}`
        : "score=NA";
    const sourceText = (pathItem.sources || []).length > 0 ? pathItem.sources.join("，") : "未知来源";
    meta.textContent = `${scoreText} | source=${sourceText}`;

    item.appendChild(main);
    item.appendChild(meta);
    pathsList.appendChild(item);
  });
}

function appendEvidenceDocs(docs) {
  clearList(docsList);
  const hasDocs = Array.isArray(docs) && docs.length > 0;
  toggleBlock(docsBlock, hasDocs);
  if (!hasDocs) {
    return;
  }

  docs.forEach((doc, index) => {
    const item = document.createElement("li");
    item.className = "doc-item";

    const fold = document.createElement("details");
    const title = document.createElement("summary");
    title.className = "doc-summary";
    title.textContent = `证据 ${index + 1}`;

    const body = document.createElement("div");
    body.className = "doc-body";

    const text = document.createElement("p");
    text.className = "doc-text";
    text.textContent = doc.content || "";

    const meta = document.createElement("p");
    meta.className = "doc-meta";
    const source = doc.metadata && doc.metadata.source ? `source=${doc.metadata.source}` : "source=未知";

    let ranking = "";
    if (typeof doc.hybrid_score === "number") {
      ranking = ` | hybrid=${doc.hybrid_score.toFixed(4)}`;
    } else if (typeof doc.rerank_score === "number") {
      ranking = ` | rerank=${doc.rerank_score.toFixed(4)}`;
    } else if (typeof doc.distance === "number") {
      ranking = ` | distance=${doc.distance.toFixed(4)}`;
    }

    meta.textContent = source + ranking;
    body.appendChild(text);
    body.appendChild(meta);
    fold.appendChild(title);
    fold.appendChild(body);
    item.appendChild(fold);
    docsList.appendChild(item);
  });
}

function renderResponse(data) {
  const payload = data || {};
  badgeMode.textContent = `模式：${modeText[payload.mode] || payload.mode || "-"}`;
  badgeCategory.textContent = `分类：${payload.question_category || "-"}`;
  appendEntityChips(payload.query_entities || []);
  appendPathEvidence(payload.paths || []);
  appendEvidenceDocs(payload.evidence_docs || []);
}

function selectMessage(messageId) {
  const message = messages.find((item) => item.id === messageId);
  if (!message || message.role !== "assistant" || !message.data) {
    return;
  }

  renderResponse(message.data);
  setDetailsOpen(true);
}

function createQuickButtons() {
  quickQuestionBox.innerHTML = "";
  quickQuestions.forEach((question) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "quick-btn";
    button.textContent = question;
    button.addEventListener("click", () => {
      questionInput.value = question;
      questionInput.focus();
    });
    quickQuestionBox.appendChild(button);
  });
}

async function readEventStream(response, onEvent) {
  if (!response.body) {
    throw new Error("浏览器不支持流式响应");
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder("utf-8");
  let buffer = "";

  while (true) {
    const { value, done } = await reader.read();
    if (done) {
      break;
    }

    buffer += decoder.decode(value, { stream: true });
    const blocks = buffer.split("\n\n");
    buffer = blocks.pop() || "";

    blocks.forEach((block) => {
      const payload = block
        .split("\n")
        .filter((line) => line.startsWith("data: "))
        .map((line) => line.slice(6))
        .join("\n");
      if (payload) {
        onEvent(JSON.parse(payload));
      }
    });
  }
}

async function handleSubmit(event) {
  event.preventDefault();

  const question = questionInput.value.trim();
  if (!question) {
    questionInput.focus();
    return;
  }

  setLoading(true);
  setSettingsOpen(false);

  createMessage("user", question, { createdAt: Date.now() });
  const assistantPlaceholder = createMessage("assistant", "正在生成回答...", {
    status: "loading",
    createdAt: Date.now(),
  });
  renderMessages();

  questionInput.value = "";
  questionInput.focus();

  try {
    const response = await fetch("/api/chat/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        question,
        mode: modeSelect.value,
        top_k: Number(topKInput.value),
      }),
    });

    if (!response.ok) {
      throw new Error(await response.text());
    }

    let finalPayload = null;
    let streamedAnswer = "";

    await readEventStream(response, (eventPayload) => {
      if (!eventPayload || !eventPayload.type) {
        return;
      }

      if (eventPayload.type === "token") {
        streamedAnswer += eventPayload.delta || "";
        updateMessageStatus(assistantPlaceholder.id, {
          content: streamedAnswer || "正在生成回答...",
          status: "loading",
        });
        updateRenderedMessageContent(
          assistantPlaceholder.id,
          streamedAnswer || "正在生成回答...",
          true,
        );
        return;
      }

      if (eventPayload.type === "done") {
        finalPayload = eventPayload.payload || null;
        updateMessageStatus(assistantPlaceholder.id, {
          content: finalPayload && finalPayload.answer
            ? finalPayload.answer.trim()
            : streamedAnswer.trim() || "未返回有效回答",
          status: "done",
          createdAt: Date.now(),
          data: finalPayload,
        });
        renderMessages();
        if (finalPayload) {
          renderResponse(finalPayload);
          setDetailsOpen(true);
        }
      }
    });
  } catch (error) {
    const message = error instanceof Error ? error.message : "出现未知错误";
    updateMessageStatus(assistantPlaceholder.id, {
      content: `请求失败：${message}`,
      status: "done",
      createdAt: Date.now(),
      data: null,
    });
    renderMessages();
  } finally {
    setLoading(false);
  }
}

function init() {
  createQuickButtons();
  topKOutput.value = topKInput.value;
  renderMessages();
  setDetailsOpen(false);

  topKInput.addEventListener("input", () => {
    topKOutput.value = topKInput.value;
  });

  settingsBtn.addEventListener("click", () => {
    setSettingsOpen(!isSettingsOpen);
  });

  detailsToggle.addEventListener("click", () => {
    setDetailsOpen(false);
  });

  document.addEventListener("click", closeSettingsOnOutsideClick);
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape") {
      setSettingsOpen(false);
    }
  });

  form.addEventListener("submit", handleSubmit);
}

init();

