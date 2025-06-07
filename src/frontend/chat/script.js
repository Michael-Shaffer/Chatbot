document.addEventListener('DOMContentLoaded', () => {
    const chatForm = document.getElementById('chat-form');
    const messageInput = document.getElementById('message-input');
    const chatMessages = document.getElementById('chat-messages');

    // --- Event Listeners ---
    chatForm.addEventListener('submit', handleFormSubmit);
    chatMessages.addEventListener('click', handleChatActions);

    // --- Functions ---
    function handleFormSubmit(e) {
        e.preventDefault();
        const messageText = messageInput.value.trim();

        if (messageText) {
            addMessage(messageText, 'user');
            messageInput.value = '';
            messageInput.focus();
            setTimeout(simulateBotResponse, 1200);
        }
    }

    function handleChatActions(e) {
        const button = e.target.closest('.copy-button');
        if (button) {
            const messageContent = button.closest('.message-content');
            const textToCopy = messageContent.querySelector('p').textContent;
            navigator.clipboard.writeText(textToCopy).then(() => {
                const originalContent = button.innerHTML;
                button.innerHTML = `<span class="copy-feedback">Copied!</span>`;
                setTimeout(() => {
                    button.innerHTML = originalContent;
                }, 1500);
            }).catch(err => {
                console.error('Failed to copy text: ', err);
            });
        }
    }

    function simulateBotResponse() {
        const activeDoc = document.querySelector('.document-item.active').textContent;
        const response = `Thinking... based on my analysis of document ${activeDoc}, the data suggests the following course of action. This is a placeholder response demonstrating the final UI.`;
        addMessage(response, 'bot');
    }

    function addMessage(text, type) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${type}-message`;

        const messageContentDiv = document.createElement('div');
        messageContentDiv.className = 'message-content';

        const p = document.createElement('p');
        p.textContent = text;
        messageContentDiv.appendChild(p);

        if (type === 'bot') {
            const avatar = document.createElement('div');
            avatar.className = 'bot-avatar';
            avatar.textContent = 'P';
            messageDiv.appendChild(avatar);
        } else {
            // Add a placeholder div for alignment on user messages, or an actual avatar if you have one
            const userAvatarPlaceholder = document.createElement('div');
            userAvatarPlaceholder.style.width = '32px'; // Match bot avatar width
            userAvatarPlaceholder.style.flexShrink = '0';
            messageDiv.appendChild(userAvatarPlaceholder);
        }

        messageDiv.appendChild(messageContentDiv);
        
        // MODIFIED: Only add action buttons for user messages
        if (type === 'user') {
            const actionsDiv = document.createElement('div');
            actionsDiv.className = 'message-actions';
            actionsDiv.innerHTML = `
                <button class="action-button copy-button" title="Copy text">
                    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor">
                        <path d="M7 3.5A1.5 1.5 0 018.5 2h3.879a1.5 1.5 0 011.06.44l3.122 3.12A1.5 1.5 0 0117 6.622V12.5a1.5 1.5 0 01-1.5 1.5h-1a.75.75 0 000 1.5h1A3 3 0 0017 12.5V6.622a3 3 0 00-.879-2.121L12.999.378A3 3 0 0010.879 0H8.5A3 3 0 005.5 3v1.5a.75.75 0 001.5 0v-1z" />
                        <path d="M10.125 10.125a.75.75 0 00-1.06 1.06l3 3a.75.75 0 001.06-1.06l-3-3z" />
                        <path d="M9.875 14.125a.75.75 0 00-1.06-1.06l-3 3a.75.75 0 001.06 1.06l3-3z" />
                        <path d="M3 9.5a1.5 1.5 0 011.5-1.5h1.879a1.5 1.5 0 011.06.44l3.122 3.12A1.5 1.5 0 0111 12.622V18.5a1.5 1.5 0 01-1.5 1.5h-5A1.5 1.5 0 013 18.5v-9z" />
                    </svg>
                </button>
            `;
            messageContentDiv.appendChild(actionsDiv);
        }

        chatMessages.appendChild(messageDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }
});
