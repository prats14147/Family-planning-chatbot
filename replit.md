# FamilyCare - Family Planning Chatbot

## Overview

FamilyCare is a bilingual (English/Nepali) web-based chatbot application that provides family planning information and guidance. The application serves as an educational resource, offering information about various family planning methods, benefits, safety, and services through an interactive conversational interface. Built with Flask, the chatbot uses a JSON-based knowledge base for static responses and includes provisions for fetching live health data when users ask about recent updates or news.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend Architecture

**Technology Stack:**
- Vanilla JavaScript for client-side interactions
- HTML5 with Jinja2 templating (Flask templates)
- CSS3 with custom styling and gradients
- Font Awesome icons for visual elements
- Google Fonts (Poppins) for typography

**Design Pattern:**
- Single Page Application (SPA) style for the chat interface
- Multi-page structure with a landing page (index.html) and chat page (chat.html)
- Component-based UI with reusable elements (navbar, message components)

**Key Features:**
- Language switching capability (English/Nepali) without page reload
- Real-time message rendering with avatar icons
- Responsive design with pastel color scheme (pink, purple, lavender, blue)
- Sticky navigation bar with smooth transitions and hover effects
- Hero section with gradient circular icon and animated "Try Chatbot" button
- About Us and Contact Us sections with contact information
- Social media icons in footer (Facebook, Instagram, Twitter)
- Smooth animations including floating effects and slide-in transitions

**Rationale:** Vanilla JavaScript was chosen for simplicity and minimal dependencies. The soft, pastel design with rounded corners and shadows creates a welcoming, accessible interface appropriate for healthcare information. The bilingual support addresses the need to serve both English and Nepali-speaking users.

### Backend Architecture

**Technology Stack:**
- Flask (Python web framework)
- JSON-based data storage for chatbot responses
- RESTful API design pattern

**Core Components:**

1. **Route Handlers:**
   - `/` - Landing page
   - `/chat` - Chat interface page
   - `/get_response` (POST) - Message processing endpoint

2. **Message Processing Logic:**
   - Keyword detection for live data triggers (latest, recent, update, new, current + Nepali equivalents)
   - Fuzzy matching capability (using difflib.get_close_matches) for handling user input variations
   - Language-aware response selection
   - Fallback mechanism: live data → static knowledge base → error message

3. **Data Structure:**
   - JSON knowledge base organized by language (english/nepali)
   - Key-value pairs mapping user queries to responses
   - Supports both exact matches and close matches for flexibility

**Design Decisions:**
- **Flask over Django:** Chosen for lightweight nature and simplicity, appropriate for a focused chatbot application without complex data models
- **JSON storage over database:** Selected for quick prototyping, easy editing of responses, and no database setup overhead. Suitable for static knowledge base content.
- **Stateless API:** Each request is independent, simplifying scaling and maintenance
- **Language parameter in requests:** Allows dynamic language switching without session management

**Alternatives Considered:**
- Database storage was considered but deemed unnecessary for static Q&A content
- WebSocket real-time communication was considered but REST API suffices for current interaction pattern

### Data Storage

**Current Implementation:**
- `chatbot_data.json` - Static knowledge base containing predefined Q&A pairs in English and Nepali
- File-based storage loaded into memory on application start
- No persistent user data storage or conversation history

**Structure:**
```json
{
  "english": { "question": "answer" },
  "nepali": { "question": "answer" }
}
```

**Rationale:** File-based JSON storage provides simplicity, easy manual updates, and sufficient performance for read-heavy chatbot operations. No database overhead or schema management required.

**Future Consideration:** The architecture includes hooks for live data fetching (`fetch_live_health_data` function), indicating planned expansion to dynamic data sources.

### Authentication & Authorization

**Current State:** 
- UI includes Sign Up and Login buttons in the navbar
- No authentication implementation in current codebase
- All chatbot functionality is publicly accessible

**Design Note:** Authentication is planned but not yet implemented. The frontend structure anticipates future integration of user accounts.

## External Dependencies

### Third-Party Libraries & Services

**Frontend Dependencies:**
- **Font Awesome 6.4.0** (CDN) - Icon library for UI elements (heart logo, user avatars, robot icons)
- **Google Fonts (Poppins)** (CDN) - Typography with weights: 300, 400, 600, 700

**Backend Dependencies:**
- **Flask** - Web framework for routing and request handling
- **requests** - HTTP library for fetching live health data from external APIs

### External API Integration

**Live Health Data Service:**
- Function `fetch_live_health_data(user_message, language)` is fully implemented
- Triggered by keywords: ['latest', 'recent', 'update', 'new', 'current'] (English) and ['नवीनतम', 'हालै', 'ताजा', 'नयाँ', 'हालको'] (Nepali)
- Uses MyHealthFinder API (health.gov) to fetch real-time health information
- Provides intelligent fallback messages directing users to WHO and local health departments when API is unavailable
- Returns bilingual responses with proper source attribution ('From live data source' or 'Recommended resources')

**Implementation Status:** External API integration is fully implemented and functional. The system detects live data keywords in both languages and fetches health information from the MyHealthFinder public API, with graceful fallbacks for error scenarios.

### Browser APIs
- Local DOM manipulation
- Fetch API (implied for AJAX requests to `/get_response`)
- CSS3 features (gradients, transitions, flexbox)

### Development Tools
- Python 3.x runtime environment
- Flask development server
- No build tools or bundlers (vanilla JavaScript/CSS)