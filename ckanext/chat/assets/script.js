ckan.module("chat-module", function ($, _) {
  "use strict";
  _ = _ || window._; // use underscore if available
  var timeline = [];

  // Private helper: Render Markdown and sanitize output
  function renderMarkdown(content) {
    var cleanHtml = "";
    if (Array.isArray(content)) {
      cleanHtml = content
        .map(function (item) {
          // Überprüfe, ob das Item ein Objekt ist
          if (typeof item === "object" && item !== null) {
            console.log("The item is of type object:", item);
            // Hier nehmen wir an, dass das Objekt eine `text`-Eigenschaft hat
            item = item.text || ""; // Fallback auf leeren String, falls `text` nicht existiert
          }
          // Stelle sicher, dass item ein String ist
          if (typeof item !== "string") {
            console.error("Item is not a string:", item);
            return ""; // Rückgabe eines leeren Strings, wenn der Input kein String ist.
          }

          var rawHtml = marked.parse(item);
          return DOMPurify.sanitize(rawHtml, {
            ALLOWED_TAGS: [
              "p",
              "pre",
              "code",
              "span",
              "div",
              "br",
              "strong",
              "em",
              "ul",
              "ol",
              "li",
              "a",
            ],
            ALLOWED_ATTR: ["class", "href"],
          });
        })
        .join("");
    } else if (content) {
      // Hier behandeln wir den Fall, wenn `content` kein Array ist
      if (typeof content === "object" && content !== null) {
        console.log("The content is of type object:", content);
        content = content.text || ""; // Fallback auf leeren String, falls `text` nicht existiert
      }

      // Stelle sicher, dass content ein String ist
      if (typeof content !== "string") {
        console.error("Content is not a string:", content);
        return ""; // Rückgabe eines leeren Strings, wenn der Input kein String ist.
      }

      var rawHtml = marked.parse(content);
      cleanHtml = DOMPurify.sanitize(rawHtml, {
        ALLOWED_TAGS: [
          "p",
          "pre",
          "code",
          "span",
          "div",
          "br",
          "strong",
          "em",
          "ul",
          "ol",
          "li",
          "a",
        ],
        ALLOWED_ATTR: ["class", "href"],
      });
    }
    return cleanHtml;
  }
  function copyToClipboard(text, button) {
    if (navigator.clipboard) {
      navigator.clipboard
        .writeText(text)
        .then(function () {
          button.html('<i class="fas fa-check"></i>');
          setTimeout(function () {
            button.html('<i class="fas fa-copy"></i>');
          }, 2000);
        })
        .catch(function (err) {
          console.error("Failed to copy: ", err);
        });
    } else {
      // Fallback for unsupported browsers
      var $tempInput = $("<textarea>");
      $("body").append($tempInput);
      $tempInput.val(text).select();

      try {
        document.execCommand("copy");
        button.html('<i class="fas fa-check"></i>');
        setTimeout(function () {
          button.html('<i class="fas fa-copy"></i>');
        }, 2000);
      } catch (err) {
        console.error("Failed to copy using fallback method: ", err);
      }

      $tempInput.remove();
    }
  }
  // Private helper: Convert timestamp strings to ISO format recursively
  function convertTimestampsToISO(data) {
    if (Array.isArray(data)) {
      return data.map(convertTimestampsToISO);
    } else if (typeof data === "object" && data !== null) {
      return Object.keys(data).reduce(function (acc, key) {
        acc[key] = convertTimestampsToISO(data[key]);
        if (key === "timestamp" && typeof acc[key] === "string") {
          acc[key] = new Date(acc[key]).toISOString();
        }
        return acc;
      }, {});
    }
    return data;
  }

  return {
    options: {
      debug: false,
    },
    currentChatLabel: "Current Chat",

    // Called automatically when the module is instantiated
    initialize: function () {
      this.abortController = null;
      this.bindUI();
      this.loadPreviousChats();
      this.loadChat();
      if (this.options.debug) {
        console.log("Chat module initialized");
      }
      window.sendMessage = this.sendMessage.bind(this); // Bind sendMessage globally
    },

    // Bind all UI events within the module container and globally for sidebar elements
    bindUI: function () {
      var self = this;
      // Bind click events within the chat container
      this.el.find("#sendButton").on("click", function () {
        self.sendMessage();
      });
      this.el.find("#deleteChatButton").on("click", function () {
        self.deleteChat();
      });
      this.el.find("#newChatButton").on("click", function () {
        self.newChat();
      });
      this.el.find("#regenerateButton").on("click", function () {
        self.regenerateFailedMessage();
      });
      this.el.find("#abortButton").on("click", function () {
        self.abortStream();
      });
      // Bind keydown event for the user input textarea
      this.el.find("#userInput").on("keydown", function (e) {
        self.handleKeyDown(e);
      });
      $("#researchToggle").on("click", function () {
        const checkbox = $(this).find('input[type="checkbox"]');
        checkbox.prop("checked", !checkbox.prop("checked")); // Toggle den Zustand der Checkbox
        $(this).toggleClass("active", checkbox.prop("checked")); // Aktiviere den aktiven Stil, wenn die Checkbox wahr ist
      });
      // Since the sidebar is rendered outside the module container, bind using a global selector
      $("#chatList").on("click", "li", function () {
        var index = $(this).index();
        self.loadChat(index);
      });
    },
    addCopyButtonsToBotAnswers: function () {
      var self = this; // Store the context

      this.el.find(".bot-answer").each(function () {
        var botAnswer = $(this);

        // Check if a copy button already exists
        if (botAnswer.find(".copy-button").length > 0) return;

        // Create the copy button
        var copyButton = $(
          '<button class="copy-button"><i class="fas fa-copy"></i></button>',
        );

        // Event handler for the copy button
        copyButton.on("click", () => {
          // Extract the text from the .text class within the bot-answer
          var textToCopy = botAnswer.find(".text").text().trim();

          copyToClipboard(textToCopy, copyButton); // Call the function directly
        });

        // Position the copy button
        botAnswer.css("position", "relative");
        copyButton.css({ position: "absolute", top: "5px", right: "5px" });
        botAnswer.append(copyButton);
      });
    },

    addCopyButtonsToCodeBlocks: function () {
      var self = this; // Store the context
      this.el.find("pre code").each(function () {
        var codeBlock = $(this);
        if (codeBlock.parent().find(".copy-button").length > 0) return;

        var copyButton = $(
          '<button class="copy-button"><i class="fas fa-copy"></i></button>',
        );

        copyButton.on("click", () => {
          // Copy the text from the code block
          var textToCopy = codeBlock.text().trim();
          copyToClipboard(textToCopy, copyButton); // Call the function directly
        });

        var preElement = codeBlock.parent();
        preElement.css("position", "relative");
        copyButton.css({ position: "absolute", top: "5px", right: "5px" });
        preElement.append(copyButton);
      });
    },

    // Handler for keydown on the textarea
    handleKeyDown: function (event) {
      if (event.key === "Enter" && event.shiftKey) {
        var textarea = event.target;
        var start = textarea.selectionStart;
        var end = textarea.selectionEnd;
        textarea.value =
          textarea.value.substring(0, start) +
          "\n" +
          textarea.value.substring(end);
        textarea.selectionStart = textarea.selectionEnd = start + 1;
      } else if (event.key === "Enter") {
        event.preventDefault();
        this.sendMessage();
      }
    },

    // Load previous chats into the sidebar list
    loadPreviousChats: function () {
      var chatListElement = $("#chatList"); // Sidebar is outside the module container
      chatListElement.empty();
      var chats = JSON.parse(localStorage.getItem("previousChats")) || [];
      if (chats.length === 0) {
        this.newChat();
      }
      var self = this;
      chats.forEach(function (chat, index) {
        var listItem = $("<li>")
          .addClass("list-group-item list-group-item-action")
          .text(chat.title || "Chat " + (index + 1))
          .on("click", function () {
            self.loadChat(index);
          });
        chatListElement.append(listItem);
      });
    },

    // Load a specific chat based on its index in localStorage
    loadChat: function (index) {
      var chats = JSON.parse(localStorage.getItem("previousChats")) || [];
      timeline = [];
      // Check if index is empty or out of bounds
      if (
        index === undefined ||
        index === null ||
        index < 0 ||
        index >= chats.length
      ) {
        index = chats.length - 1; // Load the last chat
      }
      if (chats[index]) {
        var chat = chats[index];
        var messagesDiv = this.el.find("#chatbox");
        messagesDiv.empty();
        // Highlight the active chat
        $("#chatList li").removeClass("active"); // Remove active class from all
        $("#chatList li").eq(index).addClass("active"); // Add active class to the selected chat
        var self = this;
        chat.messages.forEach(function (msg) {
          self.appendMessage(msg);
        });
        console.log(timeline);
        self.addCopyButtonsToBotAnswers();
        self.addCopyButtonsToCodeBlocks();
        this.currentChatLabel = chat.title;
        // Render math in the body
        renderMathInElement(document.body, {
          delimiters: [
              { left: '$$', right: '$$', display: true },
              { left: '$', right: '$', display: true },
              { left: '\\(', right: '\\)', display: false },
              { left: '\\[', right: '\\]', display: true }
          ],
          throwOnError: false
      });
      }
    },

    // Retrieve chat history from localStorage and convert timestamps
    getChatHistory: function (label) {
      label = label || this.currentChatLabel || "Current Chat";
      var chats = JSON.parse(localStorage.getItem("previousChats")) || [];
      var storedData =
        (
          chats.find(function (chat) {
            return chat.title === label;
          }) || {}
        ).messages || [];
      return convertTimestampsToISO(storedData);
    },
    appendMessage: function (message) {
      var chatbox = this.el.find("#chatbox");
      var self = this;

      function createMessageHtml(userMsg, markdown, id) {
        return $(`
            <div id="${id}" class="message ${userMsg ? "user-message" : "bot-message bot-answer"}">
              <span class="col-2 chatavatar"><i class="fas fa-${userMsg ? "user" : "robot"}"></i></span>
              <div class="col-auto text">
                ${renderMarkdown(markdown)}
              </div>
            </div>
        `);
      }
      function createToolHtml(markdown, id, tool_id, tool_name, timestamp, succeeded = true) {
        // const statusClass = succeeded ? "border-success" : "border-danger";
        return $(`
          <div class="tool-call card position-relative m-3" id="${id}" data-tool-id="${tool_id}" title="Tool Call: ${tool_name}">
            <!-- Overlapping Icon -->
            <div class="tool-icon position-absolute top-0 start-50 translate-middle bg-white p-2 rounded-circle shadow" role="button" style="z-index:10;">
              <i class="fas fa-robot fa-2x text-${succeeded ? 'success' : 'danger'}"></i>
            </div>
            
            <!-- Collapsible Content -->
            <div class="collapse tool-content mt-3" id="${tool_id}">
                <h3 class="card-title">Tool Call: ${tool_name}</h5>
                <div class="card-text mb-1 small text-muted">${timestamp}</div>
                ${renderMarkdown(markdown)}
            </div>
          </div>
        `);
      }        
      function formatContent(content) {
        if (typeof content === "object" && content !== null) {
          if (Array.isArray(content)) {
            return (
              "<ul>" +
              content
                .map((item) => "<li>" + formatContent(item) + "</li>")
                .join("") +
              "</ul>"
            );
          } else {
            var html = "<ul>";
            for (var key in content) {
              if (content.hasOwnProperty(key)) {
                html +=
                  "<li>" + key + ": " + formatContent(content[key]) + "</li>";
              }
            }
            html += "</ul>";
            return html;
          }
        } else {
          return String(content);
        }
      }
      function combineParts(parts) {
        return parts
          .map((part) => {
            // Extrahiere die relevanten Eigenschaften
            const { content, args } = part;
            // Erstelle eine Markdown-Darstellung
            let markdown = "";
            if (content) {
              markdown += formatContent(content);
            }
            if (args) {
              markdown += formatContent(JSON.parse(args)); // args könnte ein JSON-String sein
            }
            return markdown;
          })
          .join("\n"); // Teile die Ergebnisse durch Zeilenumbrüche
      }
      function updateChatbox() {
        const chatbox = $("#chatbox");
        chatbox.empty(); // Clear chatbox
      
        let toolStack = null;
        let inToolGroup = false;
      
        timeline.forEach((entry, index) => {
          const { timestamp, parts, tool_call_id, tool_name } = entry;
      
          if (tool_call_id) {
            const combinedMarkdown = combineParts(entry.parts);
            const toolHtml = createToolHtml(
              combinedMarkdown,
              `timeline-${index}`,
              `tool-${tool_call_id}`,
              tool_name || "Unknown Tool",
              timestamp,
            );
      
            toolHtml.on("click", function () {
              self.toggleDetails(`tool-${tool_call_id}`);
            });
      
            if (!inToolGroup) {
              // Start a new tool stack
              toolStack = $('<div id="tool-stack" class="d-flex flex-row align-items-start flex-wrap my-3 ms-5"></div>');
              inToolGroup = true;
            }
      
            toolStack.append(toolHtml);
          } else {
            // Flush tool stack before rendering non-tool messages
            if (inToolGroup && toolStack) {
              chatbox.append(toolStack);
              toolStack = null;
              inToolGroup = false;
            }
      
            // Render normal messages
            parts.forEach((part) => {
              if (part.part_kind === "system-prompt") return;
      
              const Msg = part.part_kind === "user-prompt"
                ? createMessageHtml(true, part.content, `timeline-${index}`)
                : createMessageHtml(false, part.content, `timeline-${index}`);
      
              chatbox.append(Msg);
            });
          }
        });
      
        // Flush remaining tool stack if timeline ends with it
        if (inToolGroup && toolStack) {
          chatbox.append(toolStack);
        }
      }      
      const { timestamp, parts } = message;
      parts.forEach((part) => {
        const { tool_call_id } = part;
        // Wenn die tool_call_id existiert, kombiniere die Teile
        if (tool_call_id) {
          const existingEntry = timeline.find(
            (entry) => entry.tool_call_id === tool_call_id,
          );
          if (existingEntry) {
            // Füge den aktuellen Teil zur bestehenden Eintragung hinzu
            existingEntry.parts.push(part);
          } else {
            // Erstelle eine neue Eintragung für diesen tool_call_id
            timeline.push({
              tool_call_id: tool_call_id,
              timestamp: timestamp,
              parts: [part],
              tool_name: part.tool_name,
            });
          }
        } else {
          timeline.push({
            timestamp: timestamp,
            parts: [part],
          });
        }
      });
      // Entferne Duplikate aus den parts-Arrays
      timeline.forEach((entry) => {
        const uniqueParts = Array.from(
          new Set(entry.parts.map((part) => JSON.stringify(part))),
        ).map((part) => JSON.parse(part));
        entry.parts = uniqueParts;
      });
      // Entferne Duplikate aus den parts-Arrays basierend auf part_kind und Zeitstempel
      timeline.forEach((entry) => {
        const latestParts = {}; // Ein Objekt, um den neuesten Teil für jeden part_kind zu speichern
        entry.parts.forEach((part) => {
          const { part_kind, timestamp } = part;
          // Überprüfen, ob wir bereits einen Teil für diesen part_kind haben
          if (
            !latestParts[part_kind] ||
            new Date(timestamp) > new Date(latestParts[part_kind].timestamp)
          ) {
            latestParts[part_kind] = part; // Speichere den Teil, wenn er der neueste ist
          }
        });
        // Konvertiere das Object zurück in ein Array
        entry.parts = Object.values(latestParts);
      });

      // Sortiere die Timeline nach Timestamp
      timeline.sort((a, b) => new Date(a.timestamp) - new Date(b.timestamp));
      updateChatbox();
      chatbox.find("pre code").each(function () {
        if (!$(this).attr("data-highlighted")) {
          hljs.highlightElement(this);
          $(this).attr("data-highlighted", "true");
        }
      });
      const lastElement = chatbox.children().last()[0];
      if (lastElement) {
        lastElement.scrollIntoView({ behavior: "smooth", block: "start" });
      }
      $('[data-bs-toggle="tooltip"]').tooltip();
    },

    // Toggle collapsible details for tool messages.
    toggleDetails: function (collapseId) {
      var collapseElement = document.getElementById(collapseId);
      if (collapseElement) {
        new bootstrap.Collapse(collapseElement, { toggle: true });
      }
    },
    // Retrieve text from the last entry (used for renaming the chat)
    getLastEntryText: function (array) {
      var lastEntry = array[array.length - 1];
      if (lastEntry && lastEntry.parts && lastEntry.parts.length > 0) {
        var str = lastEntry.parts[lastEntry.parts.length - 1].content;
        return str.replace(/^"|"$/g, "").trim();
      }
      return null;
    },

    abortStream: function () {
      if (this.abortController) {
        this.abortController.abort();
        this.abortController = null;
      }
    },

    // Send a user message and then trigger a bot reply
    sendMessage: function () {
      var self = this;
      var text = this.el.find("#userInput").val();
      this.el.find("#userInput").val("");
      if (text.trim() !== "") {
        const messageObject = {
          instructions: null,
          kind: "request",
          parts: [
            {
              content: text,
              part_kind: "user-prompt",
              timestamp: new Date().toUTCString(),
            },
          ],
        };
        self.appendMessage(messageObject);
        var chatHistory = self.getChatHistory();
        var sendButton = this.el.find("#sendButton");
        var abortButton = this.el.find("#abortButton");
        var spinner = sendButton.find(".spinner-border");
        var buttonText = sendButton.find(".button-text");
        var icon = sendButton.find(".fa-paper-plane");
        sendButton.addClass("d-none");
        abortButton.removeClass("d-none");
        spinner.removeClass("d-none");
        buttonText.addClass("d-none");
        icon.addClass("d-none");

        if (!chatHistory.length) {
          $.post("/chat/ask", {
            text: "Output only a 3-word title for this question: " + text +". Return only the 3-word title as plain text, with no explanations, punctuation, or formatting.",
          })
            .done(function (data) {
              var label = self.getLastEntryText(data.response);
              self.updateChatTitle(self.currentChatLabel, label);
              self.currentChatLabel = label;

              // Sende die Bot-Nachricht nach dem Titel
              self.sendBotMessage(text, self.currentChatLabel, function () {
                spinner.addClass("d-none");
                buttonText.removeClass("d-none");
                icon.removeClass("d-none");
                sendButton.removeClass("d-none");
                abortButton.addClass("d-none");
              });
            })
            .fail(function () {
              alert("An error occurred while processing your request.");
              spinner.addClass("d-none");
              buttonText.removeClass("d-none");
              icon.removeClass("d-none");
              sendButton.removeClass("d-none");
              abortButton.addClass("d-none");
            });
        } else {
          // Wenn Chat-Historie vorhanden ist, sende die Bot-Nachricht direkt
          self.sendBotMessage(text, self.currentChatLabel, function () {
            spinner.addClass("d-none");
            buttonText.removeClass("d-none");
            icon.removeClass("d-none");
            sendButton.removeClass("d-none");
            abortButton.addClass("d-none");
          });
        }
      }
    },

    // Function to update chat title in localStorage
    updateChatTitle: function (oldLabel, newLabel) {
      var chats = JSON.parse(localStorage.getItem("previousChats")) || [];
      var chatIndex = chats.findIndex(function (chat) {
        return chat.title === oldLabel;
      });
      if (chatIndex !== -1) {
        chats[chatIndex].title = newLabel;
        localStorage.setItem("previousChats", JSON.stringify(chats));
        $("#chatList li").eq(chatIndex).text(newLabel); // Update the text of the corresponding list item
      }
    },

    // Send a request to the bot via SSE streaming and append its reply
    sendBotMessage: function (text, label, callback) {
      var history = this.getChatHistory(label);
      var research_check = $("#researchToggle")
        .find('input[type="checkbox"]')
        .prop("checked");
      var self = this;

      var chatbox = self.el.find("#chatbox");
      var statusEl = $(
        '<div class="message bot-message" id="stream-status">' +
          '<span class="col-2 chatavatar"><i class="fas fa-robot"></i></span>' +
          '<div class="col-auto text status-list">' +
            '<div class="status-line active">' +
              '<i class="fas fa-spinner fa-spin"></i> ' +
              '<span>Processing...</span>' +
            '</div>' +
          "</div>" +
        "</div>",
      );
      chatbox.append(statusEl);
      statusEl[0].scrollIntoView({ behavior: "smooth", block: "start" });

      self.abortController = new AbortController();

      var params = new URLSearchParams();
      params.append("text", text);
      params.append("history", JSON.stringify(history));
      params.append("research", research_check);

      fetch("/chat/ask/stream", {
        method: "POST",
        headers: { "Content-Type": "application/x-www-form-urlencoded" },
        body: params,
        credentials: "same-origin",
        signal: self.abortController.signal,
      })
        .then(function (response) {
          if (!response.ok) {
            throw new Error("HTTP " + response.status);
          }
          var reader = response.body.getReader();
          var decoder = new TextDecoder();
          var buffer = "";

          function processChunk() {
            reader
              .read()
              .then(function (result) {
                if (result.done) {
                  statusEl.remove();
                  self.abortController = null;
                  if (callback) callback();
                  return;
                }

                buffer += decoder.decode(result.value, { stream: true });
                var events = buffer.split("\n\n");
                buffer = events.pop();

                events.forEach(function (eventStr) {
                  if (!eventStr.trim()) return;
                  var eventType = "";
                  var dataStr = "";
                  eventStr.split("\n").forEach(function (line) {
                    if (line.startsWith("event: "))
                      eventType = line.slice(7);
                    if (line.startsWith("data: "))
                      dataStr = line.slice(6);
                  });

                  if (eventType === "status" && dataStr) {
                    try {
                      var data = JSON.parse(dataStr);
                      var list = statusEl.find(".status-list");
                      list.find(".status-line.active")
                        .removeClass("active")
                        .find("i")
                        .removeClass("fa-spinner fa-spin")
                        .addClass("fa-check");
                      list.append(
                        '<div class="status-line active">' +
                          '<i class="fas fa-spinner fa-spin"></i> ' +
                          "<span>" + $("<span>").text(data.message).html() + "</span>" +
                        "</div>"
                      );
                      statusEl[0].scrollIntoView({
                        behavior: "smooth",
                        block: "start",
                      });
                    } catch (e) {
                      console.error("Status parse error:", e);
                    }
                  } else if (eventType === "done" && dataStr) {
                    try {
                      statusEl.remove();
                      var data = JSON.parse(dataStr);
                      var chatindex = self.saveChat(data.response, label);
                      self.loadChat(chatindex);
                    } catch (e) {
                      console.error("Response parse error:", e);
                    }
                  }
                });

                processChunk();
              })
              .catch(function (err) {
                statusEl.remove();
                self.abortController = null;
                if (err.name === "AbortError") {
                  console.log("Stream aborted by user");
                } else {
                  console.error("Stream read error:", err);
                  alert("Ein Fehler ist aufgetreten: " + err);
                }
                if (callback) callback();
              });
          }

          processChunk();
        })
        .catch(function (err) {
          statusEl.remove();
          self.abortController = null;
          if (err.name === "AbortError") {
            console.log("Stream aborted by user");
          } else {
            alert("Ein Fehler ist aufgetreten: " + err);
          }
          if (callback) callback();
        });
    },

    // Save new messages to the chat history in localStorage
    saveChat: function (newMessages, label) {
      var chats = JSON.parse(localStorage.getItem("previousChats")) || [];
      var existingChatIndex = chats.findIndex(function (chat) {
        return chat.title === label;
      });
      if (existingChatIndex === -1) {
        chats.push({ title: label, messages: [] });
        existingChatIndex = chats.length - 1;
      }
      chats[existingChatIndex].messages =
        chats[existingChatIndex].messages.concat(newMessages);
      localStorage.setItem("previousChats", JSON.stringify(chats));
      return existingChatIndex;
    },

    // Regenerate the failed message (if any)
    regenerateFailedMessage: function () {
      var currentLabel = this.currentChatLabel || "Current Chat";
      // Retrieve chat history using your helper
      var chatHistory = this.getChatHistory(currentLabel);
      if (!chatHistory.length) {
        alert("No chat history available.");
        return;
      }
      // Locate the last message containing a part with "user-prompt"
      var lastUserIndex = -1;
      for (var i = chatHistory.length - 1; i >= 0; i--) {
        if (
          chatHistory[i].parts &&
          chatHistory[i].parts.some(function (part) {
            return part.part_kind === "user-prompt";
          })
        ) {
          lastUserIndex = i;
          break;
        }
      }
      if (lastUserIndex === -1) {
        alert("No user prompt found in chat history.");
        return;
      }

      // Remove the last user prompt message from the stored chat history
      var newChatHistory = chatHistory.slice(0, lastUserIndex);

      // Update localStorage with the trimmed chat history
      var chats = JSON.parse(localStorage.getItem("previousChats")) || [];
      var chatIndex = chats.findIndex(function (chat) {
        return chat.title === currentLabel;
      });
      if (chatIndex === -1) {
        alert("Current chat not found in storage.");
        return;
      }
      chats[chatIndex].messages = newChatHistory;
      localStorage.setItem("previousChats", JSON.stringify(chats));

      // Refresh the chat UI
      // this.el.find("#chatbox").empty();
      this.loadChat(chatIndex);

      // Retrieve the user prompt text from the removed message
      var lastUserMessage = chatHistory[lastUserIndex];
      var userPromptPart = lastUserMessage.parts.find(function (part) {
        return part.part_kind === "user-prompt";
      });
      var userText = userPromptPart ? String(userPromptPart.content) : "";
      if (!userText.trim()) {
        alert("User prompt text is empty.");
        return;
      }
      // Set the user input field and call sendMessage (which triggers the spinner)
      this.el.find("#userInput").val(userText);
      this.sendMessage();
    },

    // Send a file via AJAX (if needed)
    // sendFile: function () {
    //   var file = this.el.find("#fileInput").prop("files")[0];
    //   var formData = new FormData();
    //   formData.append("file", file);
    //   $.ajax({
    //     url: "/upload",
    //     type: "POST",
    //     data: formData,
    //     contentType: false,
    //     processData: false,
    //     success: (data) => {
    //       this.appendMessage("user", "Uploaded a file");
    //       this.appendMessage("bot", data.response);
    //     },
    //   });
    // },

    // Delete the current chat and update localStorage
    deleteChat: function () {
      var chats = JSON.parse(localStorage.getItem("previousChats")) || [];
      var currentLabel = this.currentChatLabel || "Current Chat";
      var chatIndex = chats.findIndex(function (chat) {
        return chat.title === currentLabel;
      });
      if (chatIndex !== -1) {
        chats.splice(chatIndex, 1);
        localStorage.setItem("previousChats", JSON.stringify(chats));
        this.loadPreviousChats();
        this.loadChat();
      }
    },

    // Function to start a new chat session
    newChat: function () {
      var chats = JSON.parse(localStorage.getItem("previousChats")) || [];
      const newchatlabel = "Current Chat";
      var chatIndex = chats.findIndex(function (chat) {
        return chat.title === newchatlabel;
      });
      if (chatIndex === -1) {
        chats.push({ title: newchatlabel, messages: [] });
      } else {
        chats[chatIndex].messages = [];
      }
      localStorage.setItem("previousChats", JSON.stringify(chats));
      this.loadPreviousChats();
      this.loadChat();
    },
  };
});
