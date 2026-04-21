const form = document.getElementById("chat-form");
const questionInput = document.getElementById("question");
const modeSelect = document.getElementById("mode");
const topKInput = document.getElementById("top-k");
const topKOutput = document.getElementById("top-k-output");
const submitButton = document.getElementById("submit-btn");
const statusText = document.getElementById("status-text");

const answerBox = document.getElementById("answer");
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

function setStatus(message, isError = false) {
  statusText.textContent = message;
  statusText.classList.toggle("error", isError);
}

function setLoading(loading) {
  submitButton.disabled = loading;
  submitButton.textContent = loading ? "问答中..." : "开始问答";
  questionInput.readOnly = loading;
  modeSelect.disabled = loading;
  topKInput.disabled = loading;
}

function createQuickButtons() {
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

    const title = document.createElement("p");
    title.className = "doc-title";
    title.textContent = `证据 ${index + 1}`;

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

    item.appendChild(title);
    item.appendChild(text);
    item.appendChild(meta);
    docsList.appendChild(item);
  });
}

function renderResponse(data) {
  answerBox.classList.remove("empty");
  answerBox.textContent = data.answer && data.answer.trim() ? data.answer.trim() : "未返回有效回答";

  badgeMode.textContent = `模式：${modeText[data.mode] || data.mode || "-"}`;
  badgeCategory.textContent = `分类：${data.question_category || "-"}`;
  badgeTime.textContent = `耗时：${typeof data.elapsed_ms === "number" ? `${data.elapsed_ms} ms` : "-"}`;

  appendEntityChips(data.query_entities || []);
  appendPathEvidence(data.paths || []);
  appendEvidenceDocs(data.evidence_docs || []);
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

    renderResponse(data);
    setStatus("回答已更新");
  } catch (error) {
    const message = error instanceof Error ? error.message : "出现未知错误";
    setStatus(`请求失败：${message}`, true);
  } finally {
    setLoading(false);
  }
}

function init() {
  createQuickButtons();
  topKOutput.value = topKInput.value;

  topKInput.addEventListener("input", () => {
    topKOutput.value = topKInput.value;
  });

  form.addEventListener("submit", handleSubmit);
}

init();
