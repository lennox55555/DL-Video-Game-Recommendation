# 🎮 GameQuest: AI-Powered Game Recommendation System

![Game Recommendations Demo](https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExM3E0bjgxNnp4OXF5Nzl5OWQ2ZnF6cTFva21wazNodm1kcXFudjl1ZCZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/euGq9pgXoOVJcVhwRB/giphy.gif)

## 🚀 Overview

GameQuest is a retro-themed game recommendation engine that uses AI to suggest video games based on your age and gaming preferences. Built with React and AWS Lambda WebSockets, this application delivers personalized game recommendations in a nostalgic 80s/90s arcade aesthetic.

## ✨ Features

- **Retro UI**: Immersive 80s/90s-inspired interface with neon colors and pixel art elements
- **Age-Based Recommendations**: Different game suggestions based on your age group
- **Dynamic Backgrounds**: Background visuals change based on selected games
- **Real-Time Updates**: WebSocket communication for instant recommendations

## 🧩 Game Categories

The system recommends games across various categories:

- **Kids** (Under 13): Family-friendly alternatives to Minecraft, Fortnite, Mario, and Pokémon
- **Teens** (13-17): Age-appropriate games similar to Minecraft, Fortnite, Zelda, and GTA
- **Adults** (18+): Mature recommendations based on Skyrim, Cyberpunk, COD, and FIFA preferences

## 🔧 Technical Implementation

### Frontend (React)
- Responsive UI with Framer Motion animations
- Themed component system with dynamic styling
- WebSocket integration for real-time recommendations

### Backend (AWS Lambda)
- Serverless functions for game recommendation logic
- WebSocket API for bidirectional communication
- Age and preference-based filtering algorithms

## 🛠️ Setup & Installation

### Prerequisites
- Node.js (v14+)
- AWS account (for Lambda & API Gateway)

### Client Setup
```bash
# Navigate to client directory
cd client

# Install dependencies
npm install

# Start development server
npm run dev
```

### Lambda Function
```bash
# Deploy Lambda function (requires AWS CLI)
cd server
npm install
npm run deploy
```

## 📊 How Recommendations Work

The system categorizes games by age appropriateness and similar gameplay mechanics:

1. User selects age and favorite games
2. Data is sent to Lambda via WebSocket
3. Algorithm filters recommendations based on:
   - Age appropriateness
   - Similar gameplay mechanics
   - Popular alternatives to selected games
4. Results are returned in real-time

## 📷 Screenshots

| User Selection | Game Recommendations | Themed Background |
|----------------|----------------------|-------------------|
| ![Selection](https://media.giphy.com/media/dMLmQfCO7lCA2gX3tw/giphy.gif) | ![Recommendations](https://media.giphy.com/media/ZeRsJnQxQKjvpHvBF3/giphy.gif) | ![Background](https://media.giphy.com/media/l378BzHA5FwWFXVSg/giphy.gif) |

## 🎲 Future Enhancements

- User profiles and recommendation history
- Machine learning integration for improved suggestions
- Game trailer and review integration
- Social sharing features

## 🔗 Credits & Acknowledgements

- Game data curated from multiple sources
- Retro design inspired by classic 80s/90s arcade aesthetics
- Built as a project for Duke University's Deep Learning course