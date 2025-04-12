// Updated src/components/BackgroundEffect.jsx for Video Game Theme
import { useEffect, useState } from 'react';
import { motion } from 'framer-motion';
import { useUser } from '../context/UserContext';

const BackgroundEffect = () => {
  const { user } = useUser();
  const { brightness, glow, decorations, primaryColor, accentColor, gameTheme } = user.backgroundSettings;
  const [gameElements, setGameElements] = useState([]);
  const [gridVisible, setGridVisible] = useState(true);

  useEffect(() => {
    // Create game-specific elements based on selected theme
    generateGameElements(gameTheme || 'default');
  }, [gameTheme]);

  const generateGameElements = (theme) => {
    let elements = [];
    
    // Create different elements based on game theme
    switch(theme.toLowerCase()) {
      case 'minecraft':
        elements = generateBlockElements();
        setGridVisible(true);
        break;
      case 'fortnite':
        elements = generateStormElements();
        setGridVisible(false);
        break;
      case 'zelda':
        elements = generateTriforceElements();
        setGridVisible(false);
        break;
      case 'mario':
        elements = generateMarioElements();
        setGridVisible(false);
        break;
      case 'pokemon':
        elements = generatePokeballElements();
        setGridVisible(false);
        break;
      case 'skyrim':
        elements = generateDragonElements();
        setGridVisible(false);
        break;
      default:
        elements = generateRetroElements();
        setGridVisible(true);
        break;
    }
    
    setGameElements(elements);
  };
  
  // Helper functions to generate different game elements
  const generateBlockElements = () => {
    return Array.from({ length: 20 }, (_, i) => ({
      id: `block-${i}`,
      type: 'block',
      size: Math.random() * 30 + 20,
      left: Math.random() * 100,
      top: Math.random() * 100,
      rotation: Math.floor(Math.random() * 4) * 90,
      opacity: Math.random() * 0.4 + 0.1
    }));
  };
  
  const generateStormElements = () => {
    return Array.from({ length: 10 }, (_, i) => ({
      id: `storm-${i}`,
      type: 'storm',
      size: Math.random() * 300 + 100,
      left: Math.random() * 100,
      top: Math.random() * 100,
      rotation: Math.random() * 360,
      opacity: Math.random() * 0.3 + 0.1
    }));
  };
  
  const generateTriforceElements = () => {
    return Array.from({ length: 8 }, (_, i) => ({
      id: `triforce-${i}`,
      type: 'triforce',
      size: Math.random() * 50 + 30,
      left: Math.random() * 100,
      top: Math.random() * 100,
      rotation: Math.random() * 360,
      opacity: Math.random() * 0.4 + 0.1
    }));
  };
  
  const generateMarioElements = () => {
    const types = ['mushroom', 'star', 'coin', 'pipe'];
    return Array.from({ length: 15 }, (_, i) => ({
      id: `mario-${i}`,
      type: types[Math.floor(Math.random() * types.length)],
      size: Math.random() * 40 + 20,
      left: Math.random() * 100,
      top: Math.random() * 100,
      rotation: Math.random() * 20 - 10,
      opacity: Math.random() * 0.5 + 0.2
    }));
  };
  
  const generatePokeballElements = () => {
    return Array.from({ length: 12 }, (_, i) => ({
      id: `pokeball-${i}`,
      type: 'pokeball',
      size: Math.random() * 40 + 20,
      left: Math.random() * 100,
      top: Math.random() * 100,
      rotation: Math.random() * 360,
      opacity: Math.random() * 0.4 + 0.1
    }));
  };
  
  const generateDragonElements = () => {
    return Array.from({ length: 5 }, (_, i) => ({
      id: `dragon-${i}`,
      type: 'dragon',
      size: Math.random() * 100 + 50,
      left: Math.random() * 100,
      top: Math.random() * 100,
      rotation: Math.random() * 30 - 15,
      opacity: Math.random() * 0.3 + 0.1
    }));
  };
  
  const generateRetroElements = () => {
    const shapes = ['square', 'circle', 'triangle', 'line'];
    return Array.from({ length: 25 }, (_, i) => ({
      id: `retro-${i}`,
      type: shapes[Math.floor(Math.random() * shapes.length)],
      size: Math.random() * 60 + 20,
      left: Math.random() * 100,
      top: Math.random() * 100,
      rotation: Math.random() * 360,
      opacity: Math.random() * 0.3 + 0.1
    }));
  };

  // Render game-specific elements
  const renderGameElement = (element) => {
    const primaryColorVal = primaryColor || 'var(--color-primary)';
    const accentColorVal = accentColor || 'var(--color-accent-1)';
    
    const baseStyle = {
      position: 'absolute',
      left: `${element.left}%`,
      top: `${element.top}%`,
      width: `${element.size}px`,
      height: `${element.size}px`,
      opacity: element.opacity,
      transform: `rotate(${element.rotation}deg)`,
      zIndex: -15,
      pointerEvents: 'none'
    };
    
    switch(element.type) {
      case 'block':
        return (
          <div key={element.id} 
            style={{
              ...baseStyle,
              backgroundColor: element.id.includes('3') ? primaryColorVal : accentColorVal,
              border: '2px solid rgba(0,0,0,0.3)',
              boxShadow: 'inset 3px 3px rgba(255,255,255,0.2), inset -3px -3px rgba(0,0,0,0.2)'
            }}
          />
        );
      case 'storm':
        return (
          <div key={element.id} 
            style={{
              ...baseStyle,
              background: `radial-gradient(circle, transparent 30%, ${primaryColorVal} 100%)`,
              borderRadius: '50%',
              filter: 'blur(40px)'
            }}
          />
        );
      case 'triforce':
        return (
          <div key={element.id} 
            style={{
              ...baseStyle,
              width: 0,
              height: 0,
              borderLeft: `${element.size/2}px solid transparent`,
              borderRight: `${element.size/2}px solid transparent`,
              borderBottom: `${element.size}px solid ${accentColorVal}`,
              backgroundColor: 'transparent'
            }}
          />
        );
      default:
        return (
          <div key={element.id} 
            style={{
              ...baseStyle,
              backgroundColor: element.id.includes('odd') ? primaryColorVal : accentColorVal,
              borderRadius: element.type === 'circle' ? '50%' : 
                           element.type === 'triangle' ? '0% 50% 50% 50%' : '0%',
              clipPath: element.type === 'triangle' ? 'polygon(50% 0%, 0% 100%, 100% 100%)' : 
                        element.type === 'line' ? 'polygon(0 40%, 100% 40%, 100% 60%, 0% 60%)' : 'none'
            }}
          />
        );
    }
  };

  return (
    <div className="background-effect">
      {/* Base gradient background */}
      <div 
        className="gradient-bg animate-gradient"
        style={{
          position: 'fixed',
          inset: 0,
          zIndex: -20,
          background: `linear-gradient(45deg, ${primaryColor || 'var(--color-primary)'}, ${accentColor || 'var(--color-accent-1)'})`,
          filter: `brightness(${brightness}%) blur(${glow * 0.1}px)`,
          opacity: 0.6
        }}
      />
      
      {/* Grid background (if enabled) */}
      {gridVisible && (
        <div 
          className="retro-grid"
          style={{
            position: 'fixed',
            inset: 0,
            zIndex: -18,
            backgroundImage: `
              linear-gradient(to right, ${primaryColor || 'var(--color-primary)'} 1px, transparent 1px),
              linear-gradient(to bottom, ${primaryColor || 'var(--color-primary)'} 1px, transparent 1px)
            `,
            backgroundSize: '40px 40px',
            opacity: 0.15
          }}
        />
      )}
      
      {/* Game theme elements */}
      {gameElements.map(element => renderGameElement(element))}

      {/* Decorations based on selected games */}
      {decorations.map((decoration) => (
        <motion.div
          key={decoration.id}
          initial={{ scale: 0, opacity: 0 }}
          animate={{ scale: 1, opacity: 0.8 }}
          transition={{ duration: 0.5 }}
          style={{
            position: 'absolute',
            left: `${decoration.position.x}%`,
            top: `${decoration.position.y}%`,
            color: decoration.color,
            fontSize: '2.5rem',
            textShadow: `0 0 10px ${primaryColor || 'var(--color-primary)'}`,
            zIndex: -5
          }}
        >
          {getIconForGame(decoration.type)}
        </motion.div>
      ))}
    </div>
  );
};

// Helper function to get appropriate icon for game
const getIconForGame = (game) => {
  const iconMap = {
    'minecraft': '⛏️',
    'fortnite': '🔫',
    'zelda': '🗡️',
    'mario': '🍄',
    'pokemon': '⚡',
    'gta': '🚗',
    'cod': '🎖️',
    'fifa': '⚽',
    'skyrim': '🐉',
    'sims': '🏠',
    'amongus': '👨‍🚀',
    'cyberpunk': '🤖',
  };

  return iconMap[game.toLowerCase().replace(/\s+/g, '')] || '🎮';
};

export default BackgroundEffect;