document.addEventListener('DOMContentLoaded', () => {
    const chatForm = document.getElementById('chat-form');
    const messageInput = document.getElementById('message-input');
    const chatMessages = document.getElementById('chat-messages');

    chatForm.addEventListener('submit', handleFormSubmit);
    chatMessages.addEventListener('click', handleChatActions);

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
        if (button && !button.classList.contains('copied')) { // Prevent re-copying
            const messageContent = button.closest('.message-content');
            const textToCopy = messageContent.querySelector('p').textContent;
            navigator.clipboard.writeText(textToCopy).then(() => {
                const originalIcon = button.innerHTML;
                button.innerHTML = `
                    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" class="check-icon">
                      <path fill-rule="evenodd" d="M16.704 4.153a.75.75 0 01.143 1.052l-8 10.5a.75.75 0 01-1.127.075l-4.5-4.5a.75.75 0 011.06-1.06l3.894 3.893 7.48-9.817a.75.75 0 011.052-.143z" clip-rule="evenodd" />
                    </svg>
                `;
                button.classList.add('copied');
                button.title = "Copied!";

                setTimeout(() => {
                    button.innerHTML = originalIcon;
                    button.classList.remove('copied');
                    button.title = "Copy text";
                }, 2000);
            }).catch(err => {
                console.error('Failed to copy text: ', err);
                button.title = "Failed to copy";
            });
        }
    }
    function simulateBotResponse() {
        const response = "Thinking... based on my knowledge of all available documents, here is a placeholder answer.";
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
            messageDiv.appendChild(avatar);

            const actionsDiv = document.createElement('div');
            actionsDiv.className = 'message-actions';
            actionsDiv.innerHTML = `
                <button class="action-button copy-button" title="Copy text">
                    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor">
                      <path d="M7 3.5A1.5 1.5 0 018.5 2h3.879a1.5 1.5 0 011.06.44l3.122 3.12A1.5 1.5 0 0117 6.622V12.5a1.5 1.5 0 01-1.5 1.5h-1a.75.75 0 000 1.5h1A3 3 0 0017 12.5V6.622a3 3 0 00-.879-2.121L12.999.378A3 3 0 0010.879 0H8.5A3 3 0 005.5 3v1.5a.75.75 0 001.5 0v-1z" />
                      <path d="M3 9.5a1.5 1.5 0 011.5-1.5h1.879a1.5 1.5 0 011.06.44l3.122 3.12A1.5 1.5 0 0111 12.622V18.5a1.5 1.5 0 01-1.5 1.5h-5A1.5 1.5 0 013 18.5v-9z" />
                    </svg>
                </button>
            `;
            messageContentDiv.appendChild(actionsDiv);

        } else {
            const userAvatarPlaceholder = document.createElement('div');
            userAvatarPlaceholder.style.width = '32px';
            userAvatarPlaceholder.style.flexShrink = '0';
        }

        messageDiv.appendChild(messageContentDiv);
        chatMessages.appendChild(messageDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }
});
