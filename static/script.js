let currentLanguage = 'english';

const langButtons = document.querySelectorAll('.lang-btn');
const chatMessages = document.getElementById('chatMessages');
const userInput = document.getElementById('userInput');
const sendBtn = document.getElementById('sendBtn');
const voiceBtn = document.getElementById('voice-command-btn');

langButtons.forEach(btn => {
    btn.addEventListener('click', () => {
        langButtons.forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        currentLanguage = btn.getAttribute('data-lang');
        
        const welcomeMessage = currentLanguage === 'english' 
            ? "Hello! Welcome to FamilyCare. I'm here to help you with family planning information. How can I assist you today?"
            : "नमस्ते! FamilyCare मा स्वागत छ। म तपाईंलाई परिवार नियोजन जानकारीमा मद्दत गर्न यहाँ छु। म तपाईंलाई कसरी सहयोग गर्न सक्छु?";
        
        chatMessages.innerHTML = `
            <div class="message bot-message">
                <div class="message-avatar">
                    <i class="fas fa-robot"></i>
                </div>
                <div class="message-content">
                    <p>${welcomeMessage}</p>
                </div>
            </div>
        `;
    });
});

function addMessage(message, isUser = false) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${isUser ? 'user-message' : 'bot-message'}`;
    
    const avatar = document.createElement('div');
    avatar.className = 'message-avatar';
    avatar.innerHTML = isUser ? '<i class="fas fa-user"></i>' : '<i class="fas fa-robot"></i>';
    
    const content = document.createElement('div');
    content.className = 'message-content';
    // content.innerHTML = `<p>${message}</p>`;
    content.innerHTML = message.replace(/\n/g, '<br>');
    
    messageDiv.appendChild(avatar);
    messageDiv.appendChild(content);
    chatMessages.appendChild(messageDiv);
    
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

function addUserMessage(message) {
    addMessage(message, true);
}

function addBotMessage(message) {
    addMessage(message, false);
}

// "Bot is typing..." indicator
function showTypingIndicator() {
    const typingDiv = document.createElement('div');
    typingDiv.className = 'message bot-message typing-indicator';
    typingDiv.innerHTML = `
        <div class="message-avatar">
            <i class="fas fa-robot"></i>
        </div>
        <div class="message-content">
            <p><em>Bot is typing<span class="dots">...</span></em></p>
        </div>
    `;
    chatMessages.appendChild(typingDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

function removeTypingIndicator() {
    const typingDiv = document.querySelector('.typing-indicator');
    if (typingDiv) typingDiv.remove();
}

async function sendMessage() {
    const message = userInput.value.trim();
    if (!message) return;
    
    addMessage(message, true);
    userInput.value = '';

    // Show "Bot is typing…" while waiting for response
    showTypingIndicator();
    
    try {
        const response = await fetch('/get_response', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                message: message,
                language: currentLanguage
            })
        });
        
        const data = await response.json();

        removeTypingIndicator();
        addMessage(data.response, false);
    } catch (error) {
        removeTypingIndicator();
        const errorMessage = currentLanguage === 'english' 
            ? "Sorry, I'm having trouble connecting. Please try again."
            : "माफ गर्नुहोस्, मलाई जडान गर्न समस्या भइरहेको छ। कृपया फेरि प्रयास गर्नुहोस्।";
        addMessage(errorMessage, false);
    }
}

sendBtn.addEventListener('click', sendMessage);
userInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') {
        sendMessage();
    }
});

// Voice recording
let isRecording = false;
let mediaRecorder;
let audioChunks = [];

voiceBtn.addEventListener('click', async () => {
    if (isRecording) {
        stopRecording();
    } else {
        await startRecording();
    }
});

async function startRecording() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        mediaRecorder = new MediaRecorder(stream);
        audioChunks = [];

        mediaRecorder.ondataavailable = (event) => {
            if (event.data.size > 0) {
                audioChunks.push(event.data);
            }
        };

        mediaRecorder.onstop = handleRecordingStop;
        mediaRecorder.start();
        isRecording = true;
        voiceBtn.classList.add('recording');
        voiceBtn.innerHTML = '<i class="fas fa-stop"></i>';
        console.log("🎙 Recording started…");
    } catch (err) {
        console.error("Microphone access denied:", err);
        addBotMessage("I couldn't access your microphone. Please check permissions.");
    }
}

function stopRecording() {
    if (!mediaRecorder || mediaRecorder.state === 'inactive') return;
    mediaRecorder.stop();
    isRecording = false;
    voiceBtn.classList.remove('recording');
    voiceBtn.innerHTML = '<i class="fas fa-microphone"></i>';
    console.log("🛑 Recording stopped.");
}

async function handleRecordingStop() {
    if (!audioChunks.length) {
        addBotMessage("I couldn't capture audio. Please try again.");
        return;
    }

    const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
    const formData = new FormData();
    formData.append('audio', audioBlob, 'voice.webm');

    showTypingIndicator();
    try {
        const response = await fetch('/voice_query', {
            method: 'POST',
            body: formData
        });

        const result = await response.json();
        removeTypingIndicator();

        if (!response.ok || result.error) {
            throw new Error(result.error || 'Voice processing failed');
        }

        if (result.user_text) {
            addUserMessage(result.user_text);
        }
        if (result.bot_reply) {
            addBotMessage(result.bot_reply);
        }
    } catch (error) {
        console.error("Voice query error:", error);
        removeTypingIndicator();
        addBotMessage("Sorry, I couldn't process the voice message. Please try again.");
    } finally {
        audioChunks = [];
    }
}
