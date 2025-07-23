// In chat/script.js

// --- Corrected function to add a message to the chat window ---
function addMessageToChat(text, sender) {
    const messageContainer = document.createElement('div');
    messageContainer.classList.add('message', `${sender}-message`);

    // Create the inner content wrapper that your CSS expects
    const contentDiv = document.createElement('div');
    contentDiv.classList.add('message-content');

    if (text) {
        contentDiv.innerHTML = text; // For user messages, just add text
    }

    messageContainer.appendChild(contentDiv);
    chatMessages.appendChild(messageContainer);
    chatMessages.scrollTop = chatMessages.scrollHeight; // Auto-scroll

    // Return the content div to be populated by the stream
    return contentDiv;
}

// --- Corrected function that fetches and processes the stream ---
async function getBotResponse(query) {
    // This now returns the inner .message-content div, ready for content
    const botContentElement = addMessageToChat('', 'bot');

    let fullBotResponse = ''; // Variable to accumulate the full response

    try {
        const response = await fetch('/api/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ query: query }),
        });

        if (!response.ok) {
            throw new Error(`Network response was not ok`);
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder();

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            
            const chunk = decoder.decode(value);
            fullBotResponse += chunk;

            // Use marked.parse() to set the innerHTML of the content div
            botContentElement.innerHTML = marked.parse(fullBotResponse);
            
            chatMessages.scrollTop = chatMessages.scrollHeight;
        }

    } catch (error) {
        console.error('Error fetching bot response:', error);
        botContentElement.innerHTML = 'Sorry, an error occurred.';
    }
}
