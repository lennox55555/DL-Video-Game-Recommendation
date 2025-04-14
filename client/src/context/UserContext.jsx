import { createContext, useState, useContext } from 'react';

const UserContext = createContext();

export const UserProvider = ({ children }) => {
  const [user, setUser] = useState({
    username: '',
    age: 18,
    modelType: '', // Deep Learning, Traditional, or Naive
    hobbies: [],
    gameRatings: {}, // Store ratings for each game
    backgroundSettings: {
      brightness: 100,
      glow: 10,
      primaryColor: 'var(--color-primary)',
      accentColor: 'var(--color-accent-1)',
      gameTheme: 'none',
      decorations: [],
    }
  });

  const updateUser = (newData) => {
    setUser(prevUser => ({
      ...prevUser,
      ...newData
    }));
  };

  const updateBackgroundSettings = (settings) => {
    setUser(prevUser => ({
      ...prevUser,
      backgroundSettings: {
        ...prevUser.backgroundSettings,
        ...settings
      }
    }));
  };

  const addHobby = (hobby) => {
    if (!user.hobbies.includes(hobby)) {
      const newHobbies = [...user.hobbies, hobby];
      setUser(prevUser => ({
        ...prevUser,
        hobbies: newHobbies,
      }));

      // Add a new decoration for this hobby
      const newDecoration = {
        id: Date.now(),
        type: hobby,
        position: {
          x: Math.random() * 80 + 10, // 10% to 90% of viewport width
          y: Math.random() * 80 + 10, // 10% to 90% of viewport height
        },
        color: getRandomColor(),
      };

      updateBackgroundSettings({
        decorations: [...user.backgroundSettings.decorations, newDecoration]
      });
    }
  };

  const removeHobby = (hobby) => {
    const newHobbies = user.hobbies.filter(h => h !== hobby);
    const newDecorations = user.backgroundSettings.decorations.filter(d => d.type !== hobby);
    
    setUser(prevUser => ({
      ...prevUser,
      hobbies: newHobbies,
      backgroundSettings: {
        ...prevUser.backgroundSettings,
        decorations: newDecorations
      }
    }));
  };

  const updateGameRating = (game, rating) => {
    setUser(prevUser => ({
      ...prevUser,
      gameRatings: {
        ...prevUser.gameRatings,
        [game]: rating
      }
    }));
  };

  const getRandomColor = () => {
    const colors = ['#8B5CF6', '#EC4899', '#06B6D4', '#10B981', '#F59E0B', '#EF4444'];
    return colors[Math.floor(Math.random() * colors.length)];
  };

  return (
    <UserContext.Provider value={{ 
      user, 
      updateUser, 
      updateBackgroundSettings,
      addHobby,
      removeHobby,
      updateGameRating
    }}>
      {children}
    </UserContext.Provider>
  );
};

export const useUser = () => useContext(UserContext);
