document.addEventListener('DOMContentLoaded', () => {
    const chatForm = document.getElementById('chat-form');
    const messageInput = document.getElementById('message-input');
    const chatMessages = document.getElementById('chat-messages');
    const stopButton = document.getElementById('stop-button');
    let chatHistory = [];
    let currentAbortController = null;
    let isRequestInProgress = false; // New flag to track request status

    function autoResize(textarea) {
        textarea.style.height = 'auto';
        const newHeight = Math.min(textarea.scrollHeight, 120);
        textarea.style.height = newHeight + 'px';
        const wrapper = textarea.closest('.message-input-wrapper');

        if (textarea.scrollHeight > 120) {
            textarea.style.overflowY = 'auto';
        } else {
            textarea.style.overflowY = 'hidden';
        }

        if (newHeight > 30) {
            wrapper.style.alignItems = 'flex-end';
        } else {
            wrapper.style.alignItems = 'center';
        }
    }

    messageInput.addEventListener('input', () => autoResize(messageInput));

    messageInput.addEventListener('keydown', function(e) {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            if (!isRequestInProgress) {
                chatForm.dispatchEvent(new Event('submit'));
            }
        }
    });

    chatForm.addEventListener('submit', handleFormSubmit);

    stopButton.addEventListener('click', () => {
        if (currentAbortController) {
            currentAbortController.abort();
            console.log('Chatbot generation stopped by user.');
            stopButton.disabled = true;
            addMessageToChat('Bot generation stopped.', 'bot-info');
        }
    });

    async function handleFormSubmit(e) {
        e.preventDefault();

        if (isRequestInProgress) {
            console.log('Request already in progress. Please wait.');
            return;
        }

        const messageText = messageInput.value.trim();
        if (!messageText) return;

        const welcomeContainer = document.querySelector('.welcome-container');
        if (welcomeContainer) {
            welcomeContainer.remove();
        }

        chatHistory.push({ role: 'user', content: messageText });
        addMessageToChat(escapeHTML(messageText), 'user');

        messageInput.value = '';
        autoResize(messageInput);

        messageInput.disabled = true;
        stopButton.disabled = false;
        isRequestInProgress = true;

        currentAbortController = new AbortController();

        await getBotResponse(chatHistory, currentAbortController.signal);
    }

    function addMessageToChat(text, sender) {
        const messageContainer = document.createElement('div');
        messageContainer.classList.add('message', `${sender}-message`);

        const contentDiv = document.createElement('div');
        contentDiv.classList.add(`${sender}-message-content`);

        if (text) {
            contentDiv.innerHTML = text;
        }

        messageContainer.appendChild(contentDiv);

        // actions row (added for bot messages later)
        const actionsDiv = document.createElement('div');
        actionsDiv.classList.add('message-actions'); // style in CSS if you like
        messageContainer.appendChild(actionsDiv);

        chatMessages.appendChild(messageContainer);
        chatMessages.scrollTop = chatMessages.scrollHeight;

        return contentDiv; // (unchanged) callers rely on this
    }

    // Basic HTML escape for user messages to avoid accidental HTML injection
    function escapeHTML(str) {
        return str.replace(/[&<>"']/g, (m) => (
            { '&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;' }[m]
        ));
    }

    // Helper function to apply highlighting
    function applyHighlighting(element) {
        element.querySelectorAll('pre code').forEach((block) => {
            hljs.highlightElement(block);
        });
    }

    // --- FLAGGING SUPPORT ---

    function attachFlagButton(botContentElement, payloadProvider) {
        const container = botContentElement.parentElement; // messageContainer
        const actionsRow = container.querySelector('.message-actions');

        const btn = document.createElement('button');
        btn.type = 'button';
        btn.className = 'flag-btn';
        btn.title = 'Flag this response';
        btn.setAttribute('aria-label', 'Flag this response');
        btn.style.display = 'inline-flex';
        btn.style.alignItems = 'center';
        btn.style.gap = '6px';
        btn.style.fontSize = '12px';
        btn.style.padding = '6px 8px';
        btn.style.border = '1px solid var(--border, #333)';
        btn.style.borderRadius = '6px';
        btn.style.background = 'transparent';
        btn.style.cursor = 'pointer';
        btn.style.opacity = '0.8';

        btn.innerHTML = `
            <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true">
                <path d="M6 3v18H4V3h2zm2 0h8l-1.5 3L18 9H8V3z"></path>
            </svg>
            <span>Flag</span>
        `;

        const statusSpan = document.createElement('span');
        statusSpan.style.fontSize = '12px';
        statusSpan.style.marginLeft = '8px';
        statusSpan.style.opacity = '0.8';

        btn.addEventListener('click', async () => {
            btn.disabled = true;
            btn.style.opacity = '0.6';
            statusSpan.textContent = 'Sending...';

            const payload = payloadProvider();

            try {
                const res = await fetch(`${window.location.origin}/api/flag`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload),
                });

                if (!res.ok) {
                    throw new Error(`Server returned ${res.status}`);
                }

                statusSpan.textContent = 'Flagged ✓';
            } catch (err) {
                console.warn('Flagging failed, saving locally.', err);
                statusSpan.textContent = 'Saved offline';

                // Save to localStorage queue
                try {
                    const key = 'polarisFlags';
                    const existing = JSON.parse(localStorage.getItem(key) || '[]');
                    existing.push(payload);
                    localStorage.setItem(key, JSON.stringify(existing));
                } catch (_) {}

                // Also offer a one-click download so you can inspect immediately
                try {
                    const blob = new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json' });
                    const url = URL.createObjectURL(blob);
                    const a = document.createElement('a');
                    a.href = url;
                    a.download = `polaris-flag-${Date.now()}.json`;
                    a.style.display = 'none';
                    document.body.appendChild(a);
                    a.click();
                    URL.revokeObjectURL(url);
                    a.remove();
                } catch (_) {}
            } finally {
                // After sending, lock button to prevent duplicates
                btn.disabled = true;
                btn.style.opacity = '0.6';
            }
        });

        actionsRow.appendChild(btn);
        actionsRow.appendChild(statusSpan);
    }

    function buildFlagPayload({ userText, botText }) {
        return {
            timestamp: new Date().toISOString(),
            path: window.location.pathname + window.location.hash,
            userText,
            botText,
            chatHistorySnapshot: chatHistory.slice(-20), // include recent context (tune as desired)
            ua: navigator.userAgent
        };
    }

    async function getBotResponse(currentChatHistory, signal) {
        const botContentElement = addMessageToChat('', 'bot'); // Create the DOM element for bot response
        let fullBotResponse = '';

        // Capture the most recent user message now, so it's paired with this bot reply
        const lastUserMsg = (() => {
            for (let i = currentChatHistory.length - 1; i >= 0; i--) {
                if (currentChatHistory[i].role === 'user') return currentChatHistory[i].content;
            }
            return '';
        })();

        try {
            const apiUrl = `${window.location.origin}/api/chat`;
            const response = await fetch(apiUrl, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    messages: currentChatHistory,
                    stream: true,
                }),
                signal: signal
            });

            if (!response.ok) {
                const errorData = await response.json().catch(() => ({ message: 'Unknown error' }));
                throw new Error(`Network response was not ok: ${response.status} - ${errorData.message || response.statusText}`);
            }

            const reader = response.body.getReader();
            const decoder = new TextDecoder();

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;

                const chunk = decoder.decode(value);
                const lines = chunk.split('\n').filter(line => line.trim() !== '');

                for (const line of lines) {
                    if (line.startsWith('data:')) {
                        const json_str = line.substring(5).trim();

                        if (json_str === '[DONE]') {
                            console.log('Stream finished by [DONE] signal.');
                            reader.releaseLock();
                            chatHistory.push({ role: 'assistant', content: fullBotResponse });
                            applyHighlighting(botContentElement);

                            // Attach flag button now that we have the final text
                            attachFlagButton(botContentElement, () => buildFlagPayload({
                                userText: lastUserMsg,
                                botText: fullBotResponse
                            }));
                            return;
                        }

                        try {
                            const data = JSON.parse(json_str);
                            if (data.choices && data.choices.length > 0) {
                                const delta = data.choices[0].delta.content || '';
                                fullBotResponse += delta;
                                botContentElement.innerHTML = marked.parse(fullBotResponse);
                                chatMessages.scrollTop = chatMessages.scrollHeight;
                            }
                        } catch (parseError) {
                            console.error('Error parsing JSON from stream:', parseError, 'Line:', json_str);
                        }
                    }
                }
            }

            // Natural end without [DONE]
            stopButton.disabled = true;
            chatHistory.push({ role: 'assistant', content: fullBotResponse });
            applyHighlighting(botContentElement);

            attachFlagButton(botContentElement, () => buildFlagPayload({
                userText: lastUserMsg,
                botText: fullBotResponse
            }));

        } catch (error) {
            if (error.name === 'AbortError') {
                console.warn('Fetch aborted by user.');
                if (fullBotResponse) {
                    botContentElement.innerHTML = marked.parse(fullBotResponse + ' *(stopped)*');
                    applyHighlighting(botContentElement);
                    chatHistory.push({ role: 'assistant', content: fullBotResponse + ' *(stopped)*' });

                    attachFlagButton(botContentElement, () => buildFlagPayload({
                        userText: lastUserMsg,
                        botText: fullBotResponse + ' *(stopped)*'
                    }));
                }
            } else {
                console.error('Error fetching bot response:', error);
                if (fullBotResponse) {
                    botContentElement.innerHTML = marked.parse(fullBotResponse + `<br><br>Sorry, an error occurred: ${error.message}`);
                    applyHighlighting(botContentElement);
                    chatHistory.push({ role: 'assistant', content: fullBotResponse + ` (Error: ${error.message})` });

                    attachFlagButton(botContentElement, () => buildFlagPayload({
                        userText: lastUserMsg,
                        botText: fullBotResponse + ` (Error: ${error.message})`
                    }));
                } else {
                    botContentElement.innerHTML = 'Sorry, an error occurred: ' + error.message;
                    chatHistory.push({ role: 'assistant', content: `Sorry, an error occurred: ${error.message}` });

                    attachFlagButton(botContentElement, () => buildFlagPayload({
                        userText: lastUserMsg,
                        botText: `Sorry, an error occurred: ${error.message}`
                    }));
                }
            }
            stopButton.disabled = true;
        } finally {
            currentAbortController = null;
            messageInput.disabled = false;
            isRequestInProgress = false;
            messageInput.focus();
        }
    }

    messageInput.focus();
});