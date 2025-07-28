document.addEventListener('DOMContentLoaded', () => {
    const chatForm = document.getElementById('chat-form');
    const messageInput = document.getElementById('message-input');
    const chatMessages = document.getElementById('chat-messages');
    
    let chatHistory = [];

    // Auto-resize textarea functionality
    function autoResize(textarea) {
        const wrapper = textarea.closest('.message-input-wrapper');
        textarea.style.height = 'auto';
        const newHeight = Math.min(textarea.scrollHeight, 120);
        textarea.style.height = newHeight + 'px';
        
        // Toggle overflow and alignment based on content
        if (textarea.scrollHeight > 120) {
            textarea.style.overflowY = 'auto';
            wrapper.style.alignItems = 'flex-end';
        } else if (newHeight > 30) {
            textarea.style.overflowY = 'hidden';
            wrapper.style.alignItems = 'flex-end';
        } else {
            textarea.style.overflowY = 'hidden';
            wrapper.style.alignItems = 'center';
        }
    }

    // Auto-resize on input
    messageInput.addEventListener('input', function() {
        autoResize(this);
    });

    // Handle Enter key (send) vs Shift+Enter (new line)
    messageInput.addEventListener('keydown', function(e) {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            chatForm.dispatchEvent(new Event('submit'));
        }
    });

    chatForm.addEventListener('submit', handleFormSubmit);

    async function handleFormSubmit(e) {
        e.preventDefault();
        const messageText = messageInput.value.trim();
        if (!messageText) return;

        // Clear input and reset height
        messageInput.value = '';
        autoResize(messageInput);

        // Visual feedback for send button
        const sendButton = chatForm.querySelector('button[type="submit"]');
        sendButton.style.animation = 'sendPulse 0.3s ease-in-out';
        setTimeout(() => {
            sendButton.style.animation = '';
        }, 300);

        chatHistory.push({ role: 'user', content: messageText });
        
        addMessageToChat(messageText, 'user');

        await getBotResponse(chatHistory);
    }

    function addMessageToChat(text, sender) {
        const messageContainer = document.createElement('div');
        messageContainer.classList.add('message', `${sender}-message`);

        const contentDiv = document.createElement('div');
        contentDiv.classList.add('message-content');

        if (text) {
            contentDiv.innerHTML = text;
        }

        messageContainer.appendChild(contentDiv);
        chatMessages.appendChild(messageContainer);
        chatMessages.scrollTop = chatMessages.scrollHeight;

        return contentDiv;
    }

    async function getBotResponse(currentChatHistory) {
        const botContentElement = addMessageToChat('', 'bot');
        
        let fullBotResponse = '';

        try {
            const response = await fetch('/api/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ messages: currentChatHistory })
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
                
                botContentElement.innerHTML = marked.parse(fullBotResponse);
                
                chatMessages.scrollTop = chatMessages.scrollHeight;
            }

            botContentElement.querySelectorAll('pre code').forEach(block => {
                hljs.highlightElement(block);
            });

            chatHistory.push({ role: 'assistant', content: fullBotResponse });
            
        } catch (error) {
            console.error('Error fetching bot response:', error);
            botContentElement.innerHTML = 'Sorry, an error occurred: ' + error.message;
        }
    }
});
