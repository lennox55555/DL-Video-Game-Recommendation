import { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import BackgroundEffect from '../components/BackgroundEffect';
import GameCard from '../components/HobbyCard'; // Keeping filename for simplicity
import BubbleButton from '../components/BubbleButton';
import { useUser } from '../context/UserContext';
import websocketService from '../services/websocketService';

const GamesPage = () => {
  const { user, addHobby, removeHobby, updateBackgroundSettings, updateGameRating, updateUser } = useUser();
  const navigate = useNavigate();
  const [selectedGameThemes, setSelectedGameThemes] = useState({});
  const [showRatings, setShowRatings] = useState(false);

  // State for available games and loading state
  const [availableGames, setAvailableGames] = useState([
    'Minecraft', 'Fortnite', 'Zelda', 'Mario',
    'Pokemon', 'GTA', 'COD', 'FIFA',
    'Skyrim'
  ]); // Start with default games to avoid empty state
  const [isLoadingGames, setIsLoadingGames] = useState(false); // Start as false since we have default games
  const [gamesError, setGamesError] = useState('');
  
  // State to store all game titles from JSON
  const [allGameTitles, setAllGameTitles] = useState([]);

  // This function is no longer needed as we're selecting random games from a large pool
  // instead of shuffling a fixed set of games
  
  // Function to get random game titles from our local JSON data
  const fetchGameOptions = () => {
    // Show loading state
    setIsLoadingGames(true);
    setGamesError('');
    
    // If we've already loaded the titles, just pick 12 random ones
    if (allGameTitles.length > 0) {
      try {
        // Pick 12 random games from our full list
        const randomGames = getRandomGames(allGameTitles, 12);
        setAvailableGames(randomGames);
        setIsLoadingGames(false);
      } catch (error) {
        console.error('Error selecting random games:', error);
        setGamesError('Error getting random games');
        setIsLoadingGames(false);
      }
      return;
    }
    
    // If we haven't loaded the titles yet, fetch them from the JSON file
    // Use the correct path that works with the base URL configuration
    fetch('./game-titles.json')
      .then(response => {
        if (!response.ok) {
          throw new Error(`Failed to fetch game titles: ${response.status}`);
        }
        return response.json();
      })
      .then(data => {
        // Store all game titles
        if (data && data.gameTitles && Array.isArray(data.gameTitles)) {
          setAllGameTitles(data.gameTitles);
          
          // Pick 12 random games for display
          const randomGames = getRandomGames(data.gameTitles, 12);
          setAvailableGames(randomGames);
        } else {
          throw new Error('Invalid game titles data');
        }
      })
      .catch(error => {
        console.error('Error loading game titles:', error);
        setGamesError('Error loading games. Using default options.');
        // Keep using the default games
      })
      .finally(() => {
        setIsLoadingGames(false);
      });
  };
  
  // Helper function to get random games from a list
  const getRandomGames = (gamesList, count) => {
    // Check if we have enough games
    if (!gamesList || gamesList.length === 0) {
      throw new Error('No games available');
    }
    
    // If we have fewer games than requested, return all of them
    if (gamesList.length <= count) {
      return [...gamesList];
    }
    
    // Get random games without duplicates
    const result = [];
    const tempList = [...gamesList];
    
    for (let i = 0; i < count; i++) {
      const randomIndex = Math.floor(Math.random() * tempList.length);
      result.push(tempList[randomIndex]);
      tempList.splice(randomIndex, 1); // Remove the selected game to avoid duplicates
    }
    
    return result;
  };

  // Redirect if no username set and load game titles
  useEffect(() => {
    if (!user.username) {
      navigate('/');
      return;
    }
    
    // Fetch game titles from JSON on mount
    fetchGameOptions();
    
    // Cleanup function
    return () => {};
    
    // Note: we're intentionally excluding fetchGameOptions from the dependency array
    // to prevent it from running on every render, as it's a complex function that
    // refers to state values that change when it runs
  }, [user.username, navigate]);

  // Refresh game options function - reuse the fetchGameOptions function
  const refreshGameOptions = () => {
    fetchGameOptions();
  };

  // Simplified game click handler to minimize state updates
  const handleGameClick = (game, colorTheme) => {
    // Wrap in single state updates to prevent refresh
    try {
      // For Naive model, only allow one game selection
      if (user.modelType === 'Naive') {
        // If clicking on an already selected game, deselect it
        if (user.hobbies.includes(game)) {
          // Remove the game and its theme
          removeHobby(game);
          
          // Update themes
          const newThemes = { ...selectedGameThemes };
          delete newThemes[game];
          setSelectedGameThemes(newThemes);
          updateBackgroundFromThemes(newThemes);
        } else {
          // For Naive model, replace current selection with the new game
          
          // For Naive model, we want to replace all current selections
          // Clear previous selections first
          const oldGames = [...user.hobbies];
          if (oldGames.length > 0) {
            // Remove all existing games first
            oldGames.forEach(oldGame => {
              if (oldGame !== game) { // Don't remove the game we're about to add
                removeHobby(oldGame);
              }
            });
          }
          
          // Add the new game (only if not already included)
          if (!user.hobbies.includes(game)) {
            addHobby(game);
          }
          
          // Set the theme for just this game
          const newThemes = { [game]: colorTheme };
          setSelectedGameThemes(newThemes);
          updateBackgroundFromThemes(newThemes);
        }
      } else {
        // Normal behavior for Deep Learning and Traditional models
        if (user.hobbies.includes(game)) {
          // If already selected, remove the game
          removeHobby(game);
          
          // Remove from themes
          const newThemes = { ...selectedGameThemes };
          delete newThemes[game];
          setSelectedGameThemes(newThemes);
          updateBackgroundFromThemes(newThemes);
          
          // Remove rating
          updateGameRating(game, 0);
        } else {
          // If not selected, add the game
          addHobby(game);
          
          // Add to themes
          const newThemes = { ...selectedGameThemes, [game]: colorTheme };
          setSelectedGameThemes(newThemes);
          updateBackgroundFromThemes(newThemes);
          
          // Set default rating
          updateGameRating(game, 5);
        }
      }
    } catch (error) {
      console.error('Error handling game click:', error);
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

  // Handle rating change for a game
  const handleRatingChange = (game, rating) => {
    updateGameRating(game, rating);
  };

  const handleContinue = () => {
    // For Deep Learning and Traditional models, show ratings after game selection
    if ((user.modelType === 'Deep Learning' || user.modelType === 'Traditional') && !showRatings && user.hobbies.length > 0) {
      setShowRatings(true);
      return;
    }
    
    // Send the user data to the WebSocket before navigating
    websocketService.sendUserData({
      username: user.username,
      // Removed age field
      modelType: user.modelType,
      games: user.hobbies, // Send the selected games to match Lambda's expected format
      gameRatings: user.gameRatings, // Send ratings for each game
      preferences: selectedGameThemes // Additional data that might be useful
    });

    console.log('Sending game preferences to server:', user.hobbies);
    console.log('Sending game ratings to server:', user.gameRatings);
    
    // Navigate to the main page where recommendations will be displayed
    navigate('/main');
  };

  // Determine if the user should be allowed to proceed based on model type
  const canProceed = () => {
    if (user.hobbies.length === 0) return false;
    
    // For Naive model, one game selection is enough
    if (user.modelType === 'Naive') return user.hobbies.length === 1;
    
    // For rating mode, check if all selected games have ratings
    if (showRatings) {
      return user.hobbies.every(game => user.gameRatings[game] > 0);
    }
    
    return true;
  };

  return (
    <div className="page">
      <BackgroundEffect />
      
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
        className="glass-effect"
        style={{ width: '100%', maxWidth: '850px', padding: '1.5rem' }}
      >
        <h1 className="gradient-text" style={{ textAlign: 'center', marginBottom: '0.5rem' }}>
          {showRatings ? 'RATE YOUR GAMES' : 'SELECT YOUR GAMES'}
        </h1>
        
        <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', marginBottom: '1.5rem' }}>
          <p style={{ textAlign: 'center', color: 'var(--color-accent-1)', fontFamily: 'VT323, monospace', fontSize: '1.2rem', margin: 0 }}>
            {showRatings 
              ? 'Rate how much you enjoy each game (1-10)' 
              : user.modelType === 'Naive' 
                ? 'Choose ONE game for recommendations' 
                : 'Choose games to get personalized recommendations'}
          </p>
          
          {/* New Options button removed */}
        </div>
        
        {!showRatings ? (
          <div className="grid grid-2 grid-sm-3 grid-md-4" style={{ marginBottom: '1.5rem', width: '100%' }}>
            {isLoadingGames ? (
              <div style={{ gridColumn: 'span 4', textAlign: 'center', padding: '2rem 0' }}>
                <p>Loading game options...</p>
              </div>
            ) : (
              <>
                {gamesError && (
                  <div style={{ gridColumn: 'span 4', textAlign: 'center', marginBottom: '1rem' }}>
                    <p style={{ color: 'red' }}>{gamesError}</p>
                  </div>
                )}
                {availableGames.map((game) => (
                  <GameCard
                    key={game}
                    game={game}
                    selected={user.hobbies.includes(game)}
                    onClick={handleGameClick}
                  />
                ))}
              </>
            )}
          </div>
        ) : (
          <div style={{ marginBottom: '2rem' }}>
            {user.hobbies.map((game) => (
              <div key={game} style={{ 
                display: 'flex', 
                alignItems: 'center', 
                marginBottom: '1rem',
                padding: '0.75rem',
                borderRadius: '0.5rem',
                background: 'rgba(255, 255, 255, 0.1)'
              }}>
                <div style={{ flexGrow: 1 }}>
                  <div style={{ fontWeight: 'bold', marginBottom: '0.25rem' }}>{game}</div>
                  <div style={{ display: 'flex', alignItems: 'center' }}>
                    <div style={{ width: '40px', textAlign: 'center' }}>
                      {user.gameRatings[game] || 0}
                    </div>
                    <input 
                      type="range" 
                      min="1" 
                      max="10" 
                      value={user.gameRatings[game] || 0} 
                      onChange={(e) => handleRatingChange(game, parseInt(e.target.value))}
                      style={{ flexGrow: 1 }}
                    />
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}
        
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          {!showRatings ? (
            <p style={{ color: 'var(--color-accent-1)', fontFamily: 'VT323, monospace', fontSize: '1.2rem' }}>
              SELECTED: <span style={{ fontWeight: 'bold', color: 'var(--color-tertiary)' }}>{user.hobbies.length}</span>
              {user.modelType === 'Naive' && user.hobbies.length > 1 && 
                <span style={{ color: 'red', marginLeft: '0.5rem' }}>
                  (Only one allowed)
                </span>
              }
            </p>
          ) : (
            <button 
              onClick={() => setShowRatings(false)}
              style={{ 
                background: 'none', 
                border: 'none', 
                color: 'var(--color-accent-1)',
                cursor: 'pointer',
                textDecoration: 'underline'
              }}
            >
              Back to Selection
            </button>
          )}
          
          <BubbleButton 
            onClick={handleContinue} 
            disabled={!canProceed()}
          >
            {showRatings ? 'SUBMIT RATINGS' : user.modelType === 'Naive' ? 'CHOOSE THIS GAME' : 'CONTINUE'}
          </BubbleButton>
        </div>
      </motion.div>
    </div>
  );
};

export default GamesPage;