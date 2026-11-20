"use strict";
const $ = (id) => document.getElementById(id);
const token = document.querySelector('meta[name="pet-token"]').content;
let state = null, busy = false, initialized = false, editingId = null, toastTimer, motionTimer;
let renderKey = "", listKey = "", lastPollError = false;
const announcedReminders = new Set();

async function api(path, body) {
  const response = await fetch(path, {method: body === undefined ? "GET" : "POST", headers: {"Content-Type": "application/json", "X-Pet-Token": token}, ...(body === undefined ? {} : {body: JSON.stringify(body)})});
  const result = await response.json();
  if (!response.ok) throw new Error(result.error || "操作失败，请重试。");
  return result;
}
function notify(text, error = false) {
  clearTimeout(toastTimer);
  $("toast").textContent = text;
  $("toast").className = "toast" + (error ? " error" : "");
  $("toast").hidden = false;
  toastTimer = setTimeout(() => { $("toast").hidden = true; }, error ? 9000 : 5000);
}
function say(text) {
  if (!$("speak").checked || !("speechSynthesis" in window)) return;
  speechSynthesis.cancel();
  const utterance = new SpeechSynthesisUtterance(text);
  utterance.lang = "zh-CN";
  utterance.rate = 1;
  speechSynthesis.speak(utterance);
}
function animate(kind, text) {
  clearTimeout(motionTimer);
  $("pet-scene").className = "pet-scene " + kind;
  $("pet-bubble").textContent = text;
  motionTimer = setTimeout(() => { $("pet-scene").className = "pet-scene"; $("pet-bubble").textContent = "今天有什么想分享的吗？"; }, 5000);
}
function element(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}
function action(text, fn) {
  const button = element("button", "", text);
  button.type = "button";
  button.addEventListener("click", () => run(fn));
  return button;
}
async function run(fn) {
  try { await fn(); } catch (error) { notify(error.message || "连接失败，请检查服务是否运行。", true); }
}
function renderMessages() {
  const key = JSON.stringify(state.messages);
  if (key === renderKey) return;
  const messages = $("messages");
  const nearBottom = messages.scrollHeight - messages.scrollTop - messages.clientHeight < 90;
  messages.replaceChildren();
  const rows = state.messages.length ? state.messages : [{role:"assistant", content: `你好呀，我是${state.settings.name}。你可以跟我聊聊，也可以把想记住的小事放在这里。`, source: "local"}];
  for (const row of rows) {
    const wrapper = element("div", "message " + row.role);
    if (row.role === "assistant") wrapper.append(element("div", "avatar", "✳"));
    const bubble = element("div", "bubble", row.content);
    if (row.role === "assistant") bubble.append(element("span", "message-source", {local_model: "本地 v6r5 候选", demo: "离线演示 · 未调用模型", local: "本地处理"}[row.source] || "本地处理"));
    wrapper.append(bubble);
    messages.append(wrapper);
  }
  renderKey = key;
  if (nearBottom || busy || !initialized) messages.scrollTop = messages.scrollHeight;
}
function renderLists() {
  const key = JSON.stringify([state.memories, state.reminders.map(r => [r.id,r.text,r.due,r.acknowledged,r.due <= state.now])]);
  if (key === listKey) return;
  listKey = key;
  $("memory-count").textContent = state.memories.length;
  $("memories").replaceChildren();
  if (!state.memories.length) $("memories").append(element("p", "empty", "还没有小记忆。\n从一件喜欢的事开始吧。"));
  for (const memory of state.memories) {
    const item = element("div", "item");
    item.append(element("p", "", memory.text));
    const buttons = element("div", "item-actions");
    buttons.append(action("修改", () => { editingId = memory.id; $("edit-text").value = memory.text; $("edit-dialog").showModal(); }), action("删除", async () => {
      if (!confirm("删除这条记忆？原聊天记录会保留，需要时可另外清空对话。")) return;
      await api("/api/memory/delete", {id: memory.id}); await refresh();
    }));
    item.append(buttons); $("memories").append(item);
  }
  $("reminders").replaceChildren();
  if (!state.reminders.length) $("reminders").append(element("p", "empty", "还没有待办提醒。\n给自己安排一次小休息。"));
  for (const reminder of state.reminders) {
    const due = reminder.due <= state.now && !reminder.acknowledged;
    const item = element("div", "item" + (due ? " due" : "") + (reminder.acknowledged ? " completed" : ""));
    item.append(element("p", "", reminder.text), element("span", due ? "due-label" : "small", (reminder.acknowledged ? "已确认 · " : due ? "到时间了 · " : "") + new Date(reminder.due * 1000).toLocaleString("zh-CN", {month:"short",day:"numeric",hour:"2-digit",minute:"2-digit"})));
    const buttons = element("div", "item-actions");
    if (due) buttons.append(action("知道了", async () => { await api("/api/reminder/ack", {id: reminder.id}); await refresh(); }));
    buttons.append(action("删除", async () => { if (!confirm("删除这条提醒？")) return; await api("/api/reminder/delete", {id: reminder.id}); await refresh(); }));
    item.append(buttons); $("reminders").append(item);
  }
}
async function refresh() {
  state = await api("/api/state");
  lastPollError = false;
  $("connection").textContent = state.model.mode === "local" ? "● 本地模型" : "○ 离线演示";
  $("connection").className = "badge" + (state.model.configured ? "" : " demo");
  $("mode-note").textContent = state.model.mode === "local" ? "对话由本机 v6r5 候选生成，内容仍可能出错；真实移动功能未启用。" : "离线演示：本地模型未加载；记忆、提醒和宠物互动仍可使用。";
  $("mode-note").className = "mode-note" + (state.model.configured ? " connected" : "");
  $("pet-name").textContent = state.settings.name;
  document.title = state.settings.name + " · 虚拟宠物";
  if (!initialized) {
    $("name").value = state.settings.name;
    $("style").value = state.settings.style;
    $("send-memory").checked = state.settings.send_memory;
  }
  renderMessages(); renderLists();
  for (const reminder of state.reminders) {
    if (!reminder.acknowledged && reminder.due <= state.now && !announcedReminders.has(reminder.id)) {
      announcedReminders.add(reminder.id);
      notify("到时间了：" + reminder.text);
      animate("happy", "到时间了：" + reminder.text);
      say("提醒你，" + reminder.text);
    }
  }
  initialized = true;
}
async function sendMessage() {
  const text = $("message").value.trim();
  if (!text || busy) return;
  if (listening) return notify("请先停止语音识别，再检查文字并发送。", true);
  busy = true;
  $("send").disabled = true; $("message").disabled = true;
  $("thinking").textContent = `${state?.settings.name || "团团"}正在想一想…`;
  $("thinking").hidden = false;
  animate("thinking", "让我想一想…");
  try {
    const result = await api("/api/chat", {text});
    $("message").value = "";
    await refresh();
    animate("happy", "我在这里。慢慢来就好。");
    say(result.reply);
  } catch (error) {
    notify(error.message, true);
    animate("", "刚才没连上，可以再试一次。输入已保留。");
  } finally {
    busy = false; $("send").disabled = false; $("message").disabled = false; $("thinking").hidden = true; $("message").focus();
  }
}
$("chat-form").addEventListener("submit", event => { event.preventDefault(); sendMessage(); });
$("message").addEventListener("keydown", event => { if (event.key === "Enter" && !event.shiftKey && !event.isComposing) { event.preventDefault(); sendMessage(); } });
document.querySelectorAll("[data-message]").forEach(button => button.addEventListener("click", () => { if (!busy) { $("message").value = button.dataset.message; $("message").focus(); } }));
$("pat").addEventListener("click", () => animate("happy", "收到一份温柔，送你一颗小心心 ♡"));
$("play").addEventListener("click", () => animate("happy", "蹦一蹦！今天也要给自己一点快乐。"));
$("rest").addEventListener("click", () => animate("resting", "眯一会儿。你也可以放松一下肩膀。"));
$("settings-form").addEventListener("submit", event => { event.preventDefault(); run(async () => { await api("/api/settings", {name: $("name").value, style: $("style").value, send_memory: $("send-memory").checked}); renderKey = ""; await refresh(); notify("陪伴方式已保存。"); }); });
$("memory-form").addEventListener("submit", event => { event.preventDefault(); run(async () => { await api("/api/memory", {text: $("memory-text").value}); $("memory-text").value = ""; await refresh(); notify("这件小事已经记住了。"); }); });
$("edit-form").addEventListener("submit", event => { event.preventDefault(); run(async () => { await api("/api/memory", {id: editingId, text: $("edit-text").value}); $("edit-dialog").close(); await refresh(); }); });
$("cancel-edit").addEventListener("click", () => $("edit-dialog").close());
function defaultReminderTime() {
  const date = new Date(Date.now() + 10 * 60000);
  date.setMinutes(date.getMinutes() - date.getTimezoneOffset());
  $("reminder-time").value = date.toISOString().slice(0,16);
}
$("reminder-form").addEventListener("submit", event => { event.preventDefault(); run(async () => { const due = new Date($("reminder-time").value).getTime() / 1000; if (!Number.isFinite(due)) throw new Error("请选择有效的提醒时间。"); await api("/api/reminder", {text: $("reminder-text").value, due}); $("reminder-text").value = ""; defaultReminderTime(); await refresh(); notify("提醒已设置，请保持页面和服务运行。"); }); });
$("clear-chat").addEventListener("click", () => run(async () => { if (busy) return notify("请等待当前回复完成。"); if (!confirm("清空本机所有聊天记录？独立保存的记忆和提醒会保留。")) return; await api("/api/messages/clear", {}); await refresh(); }));

let recognition = null, listening = false, voiceError = "", voiceReceivedResult = false, lastVoiceTranscript = "";
const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
if (SpeechRecognition) {
  recognition = new SpeechRecognition(); recognition.lang = "zh-CN"; recognition.interimResults = true;
  recognition.onstart = () => {
    listening = true;
    $("mic").textContent = "■ 停止聆听";
    $("voice-note").textContent = "正在聆听。说完后停顿，或点“停止聆听”；识别文字不会自动发送。";
  };
  recognition.onresult = event => {
    if (busy) return;
    let transcript = "", isFinal = false;
    for (let index = 0; index < event.results.length; index++) {
      const result = event.results[index];
      transcript += result[0].transcript;
      isFinal ||= result.isFinal;
    }
    transcript = transcript.trim();
    if (!transcript) return;
    voiceReceivedResult = true;
    lastVoiceTranscript = transcript;
    $("message").value = transcript;
    $("message").focus();
    $("voice-note").textContent = isFinal ? "识别完成，请检查文字后再点击发送。" : "正在识别：" + transcript;
    if (isFinal) notify("已转成文字，检查后点击发送。");
  };
  recognition.onerror = event => {
    const messages = {
      "no-speech": "没有检测到语音，请点击语音输入后尽快说话，并在句尾停顿。",
      "not-allowed": "浏览器未允许麦克风，请在地址栏的网站权限中允许后重试。",
      "service-not-allowed": "浏览器语音识别服务不可用，请用 Edge/Chrome 或改用文字输入。",
      "audio-capture": "没有找到可用麦克风，请检查设备连接。",
      "network": "浏览器语音识别服务连接失败，请检查网络后重试。"
    };
    voiceError = messages[event.error] || "语音输入未完成（" + event.error + "），可继续打字或重试。";
    $("voice-note").textContent = voiceError;
    notify(voiceError, true);
  };
  recognition.onend = () => {
    listening = false;
    $("mic").textContent = "◉ 语音输入";
    if (voiceError) $("voice-note").textContent = voiceError;
    else if (voiceReceivedResult) {
      $("message").value = lastVoiceTranscript;
      $("message").focus();
      $("voice-note").textContent = "识别完成：“" + lastVoiceTranscript + "”。请检查文字后再点击发送。";
    }
    else $("voice-note").textContent = "没有识别到文字。请重试，或直接输入。";
    $("send").disabled = false;
    voiceError = "";
  };
  $("mic").addEventListener("click", () => {
    if (busy) return;
    if (listening) return recognition.stop();
    if (!sessionStorage.getItem("pet-voice-consent")) {
      if (!confirm("语音识别可能将音频发送到浏览器厂商的在线服务。是否启用本次页面会话的语音输入？识别文字不会自动发送给本地模型。")) return;
      sessionStorage.setItem("pet-voice-consent", "yes");
    }
    try {
      voiceError = "";
      voiceReceivedResult = false;
      lastVoiceTranscript = "";
      listening = true;
      $("mic").textContent = "… 正在启动";
      $("send").disabled = true;
      $("voice-note").textContent = "正在启动语音识别…";
      recognition.start();
    } catch {
      listening = false;
      $("mic").textContent = "◉ 语音输入";
      $("send").disabled = false;
      $("voice-note").textContent = "暂时无法启用语音输入，请重试或打字。";
      notify("暂时无法启用语音输入，请重试或打字。", true);
    }
  });
} else {
  $("mic").disabled = true;
  $("mic").title = "此浏览器不支持语音识别";
  $("voice-note").textContent = "此浏览器不支持语音识别。请在 Edge/Chrome 中打开本地网页，或直接输入文字。";
}
if (!("speechSynthesis" in window)) $("speak").disabled = true;
$("speak").addEventListener("change", () => { if (!$("speak").checked && "speechSynthesis" in window) speechSynthesis.cancel(); });
defaultReminderTime();
async function poll() {
  try { await refresh(); }
  catch {
    $("connection").textContent = "● 服务未连接"; $("connection").className = "badge error";
    if (!lastPollError) notify("本地服务未连接，请确认程序仍在运行。如果刚重启过服务，请刷新页面。", true);
    lastPollError = true;
  }
  setTimeout(poll, 2000);
}
poll();
