import { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import BackgroundEffect from '../components/BackgroundEffect';
import GameCard from '../components/HobbyCard'; // Keeping filename for simplicity
import BubbleButton from '../components/BubbleButton';
import { useUser } from '../context/UserContext';
import websocketService from '../services/websocketService';

const GamesPage = () => {
  const { user, addHobby, removeHobby, updateBackgroundSettings } = useUser();
  const navigate = useNavigate();
  const [selectedGameThemes, setSelectedGameThemes] = useState({});

  // Redirect if no username set
  useEffect(() => {
    if (!user.username) {
      navigate('/');
    }
  }, [user.username, navigate]);

  const games = [
    'Minecraft', 'Fortnite', 'Zelda', 'Mario',
    'Pokemon', 'GTA', 'COD', 'FIFA',
    'Skyrim', 'Sims', 'Among Us', 'Cyberpunk'
  ];

  const handleGameClick = (game, colorTheme) => {
    // Toggle game selection
    if (user.hobbies.includes(game)) {
      removeHobby(game);
      
      // Remove this game's theme from selected themes
      const newThemes = { ...selectedGameThemes };
      delete newThemes[game];
      setSelectedGameThemes(newThemes);
      
      // Update background with remaining themes or default
      updateBackgroundFromThemes(newThemes);
    } else {
      addHobby(game);
      
      // Add this game's theme to selected themes
      const newThemes = { 
        ...selectedGameThemes,
        [game]: colorTheme 
      };
      setSelectedGameThemes(newThemes);
      
      // Update background with new theme
      updateBackgroundFromThemes(newThemes);
    }
  };

  // Function to blend multiple game themes
  const updateBackgroundFromThemes = (themes) => {
    if (Object.keys(themes).length === 0) {
      // Default theme if no games selected
      updateBackgroundSettings({
        brightness: 100,
        glow: 10,
        primaryColor: 'var(--color-primary)',
        accentColor: 'var(--color-accent-1)',
        gameTheme: 'none'
      });
      return;
    }
    
    // Get the most recently selected game theme
    const latestGame = Object.keys(themes)[Object.keys(themes).length - 1];
    const latestTheme = themes[latestGame];
    
    // Update background with this theme
    updateBackgroundSettings({
      brightness: 110,
      glow: 20,
      primaryColor: latestTheme?.primary || 'var(--color-primary)',
      accentColor: latestTheme?.accent || 'var(--color-accent-1)',
      gameTheme: latestGame
    });
  };

  const handleContinue = () => {
    // Send the user data to the WebSocket before navigating
    websocketService.sendUserData({
      username: user.username,
      age: user.age,
      games: user.hobbies, // Send the selected games to match Lambda's expected format
      preferences: selectedGameThemes // Additional data that might be useful
    });

    console.log('Sending game preferences to server:', user.hobbies);
    
    // Navigate to the main page where recommendations will be displayed
    navigate('/main');
  };

  return (
    <div className="page">
      <BackgroundEffect />
      
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
        className="glass-effect"
        style={{ width: '100%', maxWidth: '768px', padding: '2rem' }}
      >
        <h1 className="gradient-text" style={{ textAlign: 'center', marginBottom: '0.5rem' }}>
          SELECT YOUR GAMES
        </h1>
        
        <p style={{ textAlign: 'center', color: 'var(--color-accent-1)', marginBottom: '1.5rem', fontFamily: 'VT323, monospace', fontSize: '1.2rem' }}>
          Choose games to get personalized recommendations
        </p>
        
        <div className="grid grid-2 grid-sm-3 grid-md-4" style={{ marginBottom: '2rem' }}>
          {games.map((game) => (
            <GameCard
              key={game}
              game={game}
              selected={user.hobbies.includes(game)}
              onClick={handleGameClick}
            />
          ))}
        </div>
        
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <p style={{ color: 'var(--color-accent-1)', fontFamily: 'VT323, monospace', fontSize: '1.2rem' }}>
            SELECTED: <span style={{ fontWeight: 'bold', color: 'var(--color-tertiary)' }}>{user.hobbies.length}</span>
          </p>
          
          <BubbleButton 
            onClick={handleContinue} 
            disabled={user.hobbies.length === 0}
          >
            CONTINUE
          </BubbleButton>
        </div>
      </motion.div>
    </div>
  );
};

export default GamesPage;