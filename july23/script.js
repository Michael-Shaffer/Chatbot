// In chat/script.js

// Place this at the top level of your script
let chatHistory = [];

// --- Your existing sendMessage or handleFormSubmit function ---
async function sendMessage() {
    const query = userInput.value.trim();
    if (!query) return;

    // Add user message to our JS history
    chatHistory.push({ role: 'user', content: query });

    displayMessage('You', query);
    userInput.value = '';

    // --- Create a placeholder for the bot's streaming response ---
    const botMessageDiv = document.createElement('div');
    // ... (code to create and append the bot message div) ...
    chatbox.appendChild(botMessageDiv);
    
    // This will hold the final response text
    let fullResponse = '';

    try {
        const response = await fetch('/api/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            // Send the entire history
            body: JSON.stringify({ messages: chatHistory }), 
        });

        // ... (your existing code to read the stream and append chunks to the UI) ...
        // Make sure to accumulate the chunks into a variable like `fullResponse`
        while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            const chunk = decoder.decode(value);
            fullResponse += chunk;
            // ... update the UI element with the fullResponse or chunk ...
        }

        // Add the final, complete assistant response to our JS history
        chatHistory.push({ role: 'assistant', content: fullResponse });

    } catch (error) {
        console.error('Error:', error);
        // ... handle error in UI ...
    }
}
