const form = document.getElementById("chat-form");
const questionInput = document.getElementById("question");
const modeSelect = document.getElementById("mode");
const topKInput = document.getElementById("top-k");
const topKOutput = document.getElementById("top-k-output");
const submitButton = document.getElementById("submit-btn");
const statusText = document.getElementById("status-text");
const settingsBtn = document.getElementById("settings-btn");
const settingsPopover = document.getElementById("settings-popover");
const detailsPanel = document.getElementById("details-panel");
const detailsToggle = document.getElementById("details-toggle");
const detailsContent = document.getElementById("details-content");
const workspace = document.getElementById("workspace");
const chatHistory = document.getElementById("chat-history");
const chatEmpty = document.getElementById("chat-empty");
const entitiesList = document.getElementById("entities-list");
const pathsList = document.getElementById("paths-list");
const docsList = document.getElementById("docs-list");

const badgeMode = document.getElementById("badge-mode");
const badgeCategory = document.getElementById("badge-category");
const badgeTime = document.getElementById("badge-time");
const quickQuestionBox = document.getElementById("quick-questions");

const quickQuestions = [
  "十四五期间上海综合交通发展的总体思路是什么？",
  "上海在轨道交通网络建设方面提出了哪些重点行动？",
  "什么是‘多网融合’，在政策里如何定义？",
  "相比上一轮规划，本轮政策的重点变化有哪些？",
];

const modeText = {
  vector: "Vector RAG",
  pathrag: "PathRAG",
  hybrid: "Hybrid Multi-Recall",
};

const messages = [];
let isSettingsOpen = false;
let isDetailsCollapsed = false;
let activeMessageId = null;
let messageSeed = 0;

function setStatus(message, isError = false) {
  statusText.textContent = message;
  statusText.classList.toggle("error", isError);
}

function setLoading(loading) {
  submitButton.disabled = loading;
  submitButton.textContent = loading ? "问答中..." : "发送";
  modeSelect.disabled = loading;
  topKInput.disabled = loading;
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

function clearList(target) {
  target.innerHTML = "";
  target.classList.add("empty");
}

function formatTime(timestamp) {
  return new Intl.DateTimeFormat("zh-CN", {
    hour: "2-digit",
    minute: "2-digit",
  }).format(timestamp);
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

function renderMessage(message) {
  const wrapper = document.createElement("section");
  wrapper.className = `chat-message ${message.role}`;
  wrapper.dataset.messageId = message.id;

  const role = document.createElement("div");
  role.className = "message-role";
  role.textContent = message.role === "user" ? "用户" : "模型";

  const bubble = document.createElement("div");
  bubble.className = "message-bubble";
  if (message.role === "assistant") {
    bubble.classList.add("assistant-bubble");
  }
  if (message.status === "loading") {
    bubble.classList.add("loading");
  }
  bubble.textContent = message.content;

  if (message.role === "assistant" && message.status !== "loading") {
    bubble.title = "点击查看右侧详细信息";
    bubble.addEventListener("click", () => {
      selectMessage(message.id, true);
    });
  }

  const time = document.createElement("div");
  time.className = "message-time";
  time.textContent = formatTime(message.createdAt);

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

function showEmptyState() {
  chatEmpty.classList.remove("hidden");
  renderMessages();
}

function hideEmptyState() {
  chatEmpty.classList.add("hidden");
}

function setDetailsCollapsed(collapsed) {
  isDetailsCollapsed = collapsed;
  workspace.classList.toggle("details-collapsed", collapsed);
  detailsPanel.classList.toggle("hidden", collapsed);
  detailsToggle.textContent = collapsed ? "打开详情" : "关闭详情";
  detailsToggle.setAttribute("aria-expanded", String(!collapsed));
}

function setSettingsOpen(open) {
  isSettingsOpen = open;
  settingsPopover.classList.toggle("hidden", !open);
  settingsBtn.setAttribute("aria-expanded", String(open));
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

function selectMessage(messageId, fromUserAction = false) {
  const message = messages.find((item) => item.id === messageId);
  if (!message || message.role !== "assistant") {
    return;
  }

  activeMessageId = message.id;
  renderResponse(message.data, message.content, message);

  if (isDetailsCollapsed) {
    setDetailsCollapsed(false);
  }

  if (!fromUserAction) {
    scrollChatToBottom();
  }
}

function appendEntityChips(entities) {
  clearList(entitiesList);
  if (!entities || entities.length === 0) {
    return;
  }
  entitiesList.classList.remove("empty");
  entities.forEach((entity) => {
    const item = document.createElement("li");
    item.className = "chip";
    item.textContent = entity;
    entitiesList.appendChild(item);
  });
}

function appendPathEvidence(paths) {
  clearList(pathsList);
  if (!paths || paths.length === 0) {
    return;
  }
  pathsList.classList.remove("empty");
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
    const sourceText = (pathItem.sources || []).length > 0 ? pathItem.sources.join("；") : "未知来源";
    meta.textContent = `${scoreText} | source=${sourceText}`;

    item.appendChild(main);
    item.appendChild(meta);
    pathsList.appendChild(item);
  });
}

function appendEvidenceDocs(docs) {
  clearList(docsList);
  if (!docs || docs.length === 0) {
    return;
  }

  docsList.classList.remove("empty");
  docs.forEach((doc, index) => {
    const item = document.createElement("li");
    item.className = "doc-item";

    const fold = document.createElement("details");
    fold.className = "doc-fold";

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

function renderResponse(data, fallbackAnswer = "", message = null) {
  const payload = data || {};
  badgeMode.textContent = `模式：${modeText[payload.mode] || payload.mode || "-"}`;
  badgeCategory.textContent = `分类：${payload.question_category || "-"}`;
  badgeTime.textContent = `耗时：${typeof payload.elapsed_ms === "number" ? `${payload.elapsed_ms} ms` : "-"}`;

  appendEntityChips(payload.query_entities || []);
  appendPathEvidence(payload.paths || []);
  appendEvidenceDocs(payload.evidence_docs || []);
}

function updateMessageStatus(messageId, patch) {
  const message = messages.find((item) => item.id === messageId);
  if (!message) {
    return null;
  }

  Object.assign(message, patch);
  return message;
}

async function handleSubmit(event) {
  event.preventDefault();

  const question = questionInput.value.trim();
  if (!question) {
    setStatus("请输入问题后再提交", true);
    questionInput.focus();
    return;
  }

  setLoading(true);
  setStatus("正在调用后端模型，请稍候...");
  hideEmptyState();
  setSettingsOpen(false);

  const userMessage = createMessage("user", question, { createdAt: Date.now() });
  const assistantPlaceholder = createMessage("assistant", "正在生成回答...", {
    status: "loading",
    createdAt: Date.now(),
  });
  activeMessageId = assistantPlaceholder.id;
  renderMessages();
  // 清空输入框，但保持可编辑（用户可继续输入下一个问题），发送按钮已被禁用
  questionInput.value = "";
  questionInput.focus();

  try {
    const response = await fetch("/api/chat", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        question,
        mode: modeSelect.value,
        top_k: Number(topKInput.value),
      }),
    });

    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.detail || "问答请求失败");
    }

    updateMessageStatus(assistantPlaceholder.id, {
      content: data.answer && data.answer.trim() ? data.answer.trim() : "未返回有效回答",
      status: "done",
      data,
    });
    renderMessages();
    setDetailsCollapsed(false);
    renderResponse(data, assistantPlaceholder.content, assistantPlaceholder);
    setStatus("回答已更新");
  } catch (error) {
    const message = error instanceof Error ? error.message : "出现未知错误";
    updateMessageStatus(assistantPlaceholder.id, {
      content: `请求失败：${message}`,
      status: "done",
      data: null,
    });
    renderMessages();
    setStatus(`请求失败：${message}`, true);
  } finally {
    setLoading(false);
  }
}

function init() {
  createQuickButtons();
  topKOutput.value = topKInput.value;
  renderMessages();
  setDetailsCollapsed(true);

  topKInput.addEventListener("input", () => {
    topKOutput.value = topKInput.value;
  });

  settingsBtn.addEventListener("click", () => {
    setSettingsOpen(!isSettingsOpen);
  });

  detailsToggle.addEventListener("click", () => {
    setDetailsCollapsed(!isDetailsCollapsed);
    if (!isDetailsCollapsed && activeMessageId) {
      const message = messages.find((item) => item.id === activeMessageId);
      if (message && message.data) {
        renderResponse(message.data, message.content, message);
      }
    }
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
