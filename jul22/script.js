// Corrected and complete chat/script.js

document.addEventListener('DOMContentLoaded', () => {
    const chatForm = document.getElementById('chat-form');
    const messageInput = document.getElementById('message-input');
    const chatMessages = document.getElementById('chat-messages');

    // Handle form submission for sending a message
    chatForm.addEventListener('submit', handleFormSubmit);

    // --- Main function to handle sending a message ---
    async function handleFormSubmit(e) {
        e.preventDefault();
        const messageText = messageInput.value.trim();
        if (!messageText) return;

        // 1. Display the user's message immediately
        addMessageToChat(messageText, 'user');
        messageInput.value = '';

        // 2. Get the bot's streaming response
        await getBotResponse(messageText);
    }

    // --- Function to add a message to the chat window ---
    function addMessageToChat(text, sender) {
        const messageContainer = document.createElement('div');
        messageContainer.classList.add('message', `${sender}-message`);

        const senderStrong = document.createElement('strong');
        senderStrong.textContent = sender === 'user' ? 'You: ' : 'Bot: ';

        const contentP = document.createElement('p');
        contentP.textContent = text;
        
        messageContainer.appendChild(senderStrong);
        messageContainer.appendChild(contentP);

        chatMessages.appendChild(messageContainer);
        chatMessages.scrollTop = chatMessages.scrollHeight; // Auto-scroll
        return contentP; // Return the paragraph element to append stream to it
    }

    // --- Function that fetches and processes the stream from the server ---
    async function getBotResponse(query) {
        // Create a placeholder for the bot's response
        const botTextElement = addMessageToChat('', 'bot');

        try {
            const response = await fetch('/api/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ query: query }),
            });

            if (!response.ok) {
                throw new Error(`Network response was not ok: ${response.statusText}`);
            }

            // Get the reader to process the stream
            const reader = response.body.getReader();
            const decoder = new TextDecoder();

            // Read from the stream until it's finished
            while (true) {
                const { done, value } = await reader.read();
                if (done) {
                    break; // The stream has ended
                }
                // Decode the chunk of data and append it
                const chunk = decoder.decode(value);
                botTextElement.textContent += chunk;
                chatMessages.scrollTop = chatMessages.scrollHeight; // Keep scrolling
            }

        } catch (error) {
            console.error('Error fetching bot response:', error);
            botTextElement.textContent = 'Sorry, an error occurred while connecting to the bot.';
        }
    }
});
