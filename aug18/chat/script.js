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
            // Only dispatch submit if no request is in progress
            if (!isRequestInProgress) {
                chatForm.dispatchEvent(new Event('submit'));
            }
        }
    });

    chatForm.addEventListener('submit', handleFormSubmit);

    stopButton.addEventListener('click', () => {
        if (currentAbortController) {
            currentAbortController.abort(); // Abort the ongoing fetch request
            console.log('Chatbot generation stopped by user.');
            stopButton.disabled = true; // Disable stop button after stopping
            // This message is added immediately on stop, before the catch block handles the AbortError
            addMessageToChat('Bot generation stopped.', 'bot-info');
        }
    });

    async function handleFormSubmit(e) {
        e.preventDefault();

        // Prevent submission if a request is already in progress
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

        chatHistory.push({ role: 'user', content: messageText }); // Add user message to history
        addMessageToChat(messageText, 'user');

        messageInput.value = '';
        autoResize(messageInput);

        // Disable input and set flag
        messageInput.disabled = true;
        stopButton.disabled = false;
        isRequestInProgress = true;

        currentAbortController = new AbortController();

        // Pass chatHistory to getBotResponse for the LLM call
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
        chatMessages.appendChild(messageContainer);
        chatMessages.scrollTop = chatMessages.scrollHeight;

        return contentDiv;
    }

    // Helper function to apply highlighting
    function applyHighlighting(element) {
        // Find all pre > code blocks within the given element and highlight them
        element.querySelectorAll('pre code').forEach((block) => {
            hljs.highlightElement(block);
        });
    }

    async function getBotResponse(currentChatHistory, signal) {
        const botContentElement = addMessageToChat('', 'bot'); // Create the DOM element for bot response
        let fullBotResponse = '';

        try {
            const apiUrl = `${window.location.origin}/api/chat`;
            const response = await fetch(apiUrl, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    messages: currentChatHistory, // Send the full chat history
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
                            // Add bot's full response to chatHistory here
                            chatHistory.push({ role: 'assistant', content: fullBotResponse });
                            applyHighlighting(botContentElement); // Highlight the final content
                            return; // Exit the function
                        }

                        try {
                            const data = JSON.parse(json_str);
                            if (data.choices && data.choices.length > 0) {
                                const delta = data.choices[0].delta.content || '';
                                fullBotResponse += delta;
                                // Update innerHTML with parsed markdown incrementally
                                botContentElement.innerHTML = marked.parse(fullBotResponse);
                                chatMessages.scrollTop = chatMessages.scrollHeight;
                            }
                        } catch (parseError) {
                            console.error('Error parsing JSON from stream:', parseError, 'Line:', json_str);
                        }
                    }
                }
            }

            // This block is reached if the stream ends naturally (done: true) without a [DONE] signal
            stopButton.disabled = true;
            // Add bot's full response to chatHistory here as well for natural stream end
            chatHistory.push({ role: 'assistant', content: fullBotResponse });
            applyHighlighting(botContentElement); // Ensure highlighting for naturally ended stream

        } catch (error) {
            if (error.name === 'AbortError') {
                console.warn('Fetch aborted by user.');
                if (fullBotResponse) {
                    botContentElement.innerHTML = marked.parse(fullBotResponse + ' *(stopped)*');
                    applyHighlighting(botContentElement); // Highlight even if aborted mid-generation
                    chatHistory.push({ role: 'assistant', content: fullBotResponse + ' *(stopped)*' });
                }
            } else {
                console.error('Error fetching bot response:', error);
                if (fullBotResponse) {
                    botContentElement.innerHTML = marked.parse(fullBotResponse + `<br><br>Sorry, an error occurred: ${error.message}`);
                    applyHighlighting(botContentElement); // Highlight on error if partial content exists
                    chatHistory.push({ role: 'assistant', content: fullBotResponse + ` (Error: ${error.message})` });
                } else {
                    botContentElement.innerHTML = 'Sorry, an error occurred: ' + error.message;
                    chatHistory.push({ role: 'assistant', content: `Sorry, an error occurred: ${error.message}` });
                }
            }
            stopButton.disabled = true;
        } finally {
            currentAbortController = null;
            // Re-enable input and reset flag
            messageInput.disabled = false;
            isRequestInProgress = false;
            messageInput.focus(); // Bring focus back to the input
        }
    }

    messageInput.focus();
});
