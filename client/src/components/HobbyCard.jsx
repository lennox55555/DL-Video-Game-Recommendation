import { motion } from 'framer-motion';

const GameCard = ({ game, selected, onClick }) => {
  // Game icons mapping
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

  // Game color themes mapping for background effects
  const colorThemes = {
    'minecraft': { primary: '#5D7C3F', accent: '#916A3D' },
    'fortnite': { primary: '#1F48CF', accent: '#B026FF' },
    'zelda': { primary: '#2E8B57', accent: '#DAA520' },
    'mario': { primary: '#E4000F', accent: '#01ADEF' },
    'pokemon': { primary: '#FFDE00', accent: '#3B4CCA' },
    'gta': { primary: '#000000', accent: '#FF4500' },
    'cod': { primary: '#355E3B', accent: '#8B0000' },
    'fifa': { primary: '#00873C', accent: '#0033A0' },
    'skyrim': { primary: '#647687', accent: '#2E3B55' },
    'sims': { primary: '#69C0FF', accent: '#00D166' },
    'amongus': { primary: '#C51111', accent: '#132ED2' },
    'cyberpunk': { primary: '#F9E900', accent: '#00FFFF' },
  };

  const gameLower = game.toLowerCase().replace(/[^a-z0-9]/g, '');
  const icon = iconMap[gameLower] || '🎮';

  // Apply minimal formatting for extremely long titles
  const formatGameTitle = (title) => {
    // For titles longer than 25 chars, do some minimal formatting
    if (title.length > 25) {
      // Remove any parenthetical information
      const withoutParentheses = title.replace(/\s*\([^)]*\)\s*/g, ' ').trim();
      
      // Remove any "The" at the beginning
      const withoutThe = withoutParentheses.replace(/^The\s+/i, '');
      
      // If still too long, use the shorter version
      return withoutThe.length < title.length ? withoutThe : title;
    }
    return title;
  };

  return (
    <motion.div
      onClick={() => onClick(game, colorThemes[gameLower])}
      className={`card card-game ${selected ? 'selected glass-effect' : ''}`}
      whileHover={{ scale: selected ? 1.05 : 1.1 }}
      whileTap={{ scale: 0.95 }}
      style={{
        borderColor: selected ? colorThemes[gameLower]?.primary : undefined,
        boxShadow: selected ? `0 0 15px ${colorThemes[gameLower]?.accent}` : undefined
      }}
      title={game} // Show full title on hover
    >
      <div className="card-game-title">{formatGameTitle(game)}</div>
    </motion.div>
  );
};

export default GameCard;
