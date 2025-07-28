document.addEventListener('DOMContentLoaded', () => {
    const chatForm = document.getElementById('chat-form');
    const messageInput = document.getElementById('message-input');
    const chatMessages = document.getElementById('chat-messages');
    let chatHistory = [];

    // Auto-resize for the textarea
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
            chatForm.dispatchEvent(new Event('submit'));
        }
    });

    chatForm.addEventListener('submit', handleFormSubmit);

    // Main submit handler
    async function handleFormSubmit(e) {
        e.preventDefault();
        const messageText = messageInput.value.trim();
        if (!messageText) return;

        // --- KEY ADDITION: Remove welcome screen on first message ---
        const welcomeContainer = document.querySelector('.welcome-container');
        if (welcomeContainer) {
            welcomeContainer.remove();
        }
        // --- End of addition ---

        chatHistory.push({ role: 'user', content: messageText });
        addMessageToChat(messageText, 'user');
        
        messageInput.value = '';
        autoResize(messageInput);
        
        await getBotResponse(chatHistory);
    }

    // Function to add a message to the chat DOM
    function addMessageToChat(text, sender) {
        const messageContainer = document.createElement('div');
        messageContainer.classList.add('message', `${sender}-message`);

        const contentDiv = document.createElement('div');
        contentDiv.classList.add(`${sender}-message-content`);

        if (text) {
            // Use innerHTML to render potential markdown, but be cautious with untrusted content
            contentDiv.innerHTML = text; 
        }
        
        messageContainer.appendChild(contentDiv);
        chatMessages.appendChild(messageContainer);
        chatMessages.scrollTop = chatMessages.scrollHeight;
        
        return contentDiv; // Return the element where content will be streamed
    }

    // Function to get response from the bot API
    async function getBotResponse(currentChatHistory) {
        const botContentElement = addMessageToChat('', 'bot'); // Create empty bot message
        let fullBotResponse = '';

        try {
            const response = await fetch('/api/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ messages: currentChatHistory }),
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
                fullBotResponse += chunk;
                botContentElement.innerHTML = marked.parse(fullBotResponse); // Assuming you use a library like 'marked'
                
                chatMessages.scrollTop = chatMessages.scrollHeight;
            }

            // Highlight code blocks after the full message is received
            botContentElement.querySelectorAll('pre code').forEach((block) => {
                hljs.highlightElement(block);
            });

            chatHistory.push({ role: 'assistant', content: fullBotResponse });

        } catch (error) {
            console.error('Error fetching bot response:', error);
            botContentElement.innerHTML = 'Sorry, an error occurred: ' + error.message;
        }
    }

    // Initial focus on input
    messageInput.focus();
});
