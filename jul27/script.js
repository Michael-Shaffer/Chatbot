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

// Get the message input element
const messageInput = document.getElementById('message-input');

// Auto-resize on input
messageInput.addEventListener('input', function() {
    autoResize(this);
});

// Handle Enter key (send) vs Shift+Enter (new line)
messageInput.addEventListener('keydown', function(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        document.getElementById('chat-form').dispatchEvent(new Event('submit'));
    }
});

// Update your existing form submit handler
document.getElementById('chat-form').addEventListener('submit', function(e) {
    e.preventDefault();
    
    if (messageInput.value.trim()) {
        // Your existing message sending logic here
        const message = messageInput.value.trim();
        
        // Send the message (your existing code)
        // ...
        
        // Clear the input and reset height
        messageInput.value = '';
        autoResize(messageInput);
        
        // Visual feedback
        const button = this.querySelector('button');
        button.style.animation = 'sendPulse 0.3s ease-in-out';
        setTimeout(() => {
            button.style.animation = '';
        }, 300);
    }
});
