// src/services/websocketService.js
class WebSocketService {
  constructor() {
    this.socket = null;
    this.isConnected = false;
    this.messageQueue = [];
    this.callbacks = {};
  }

  connect(username) {
    return new Promise((resolve, reject) => {
      // Using the WebSocket URL from your Lambda function
      const wsUrl = 'wss://8wv7noht30.execute-api.us-east-1.amazonaws.com/prod/';
      
      this.socket = new WebSocket(wsUrl);
      this.username = username; // Store username immediately
      
      // Connection timeout handling
      const connectionTimeout = setTimeout(() => {
        if (!this.isConnected) {
          console.error('WebSocket connection timed out');
          reject(new Error('WebSocket connection timed out'));
          
          if (this.socket) {
            this.socket.close();
          }
        }
      }, 10000); // 10 second timeout
      
      this.socket.onopen = () => {
        console.log('WebSocket connected successfully');
        this.isConnected = true;
        clearTimeout(connectionTimeout);
        
        // Process any messages that were queued while disconnected
        this._processQueue();
        
        resolve();
      };
      
      this.socket.onclose = (event) => {
        console.log(`WebSocket disconnected: Code ${event.code}, Reason: ${event.reason || 'No reason provided'}`);
        this.isConnected = false;
        clearTimeout(connectionTimeout);
        
        // If not a normal closure, attempt reconnection after delay
        if (event.code !== 1000 && event.code !== 1001) {
          console.log('Abnormal closure, attempting reconnect in 3 seconds...');
          setTimeout(() => {
            if (this.username && !this.isConnected) {
              console.log('Attempting to reconnect WebSocket...');
              this.connect(this.username)
                .then(() => console.log('Reconnected successfully'))
                .catch(err => console.error('Reconnection failed:', err));
            }
          }, 3000);
        }
      };
      
      this.socket.onerror = (error) => {
        console.error('WebSocket error:', error);
        clearTimeout(connectionTimeout);
        reject(error);
      };
      
      // Track message rates to avoid flooding
      let messageCount = 0;
      let lastMessageTime = Date.now();
      
      this.socket.onmessage = (event) => {
        try {
          // Track message rate
          messageCount++;
          const now = Date.now();
          // Reset count every 5 seconds
          if (now - lastMessageTime > 5000) {
            messageCount = 1;
            lastMessageTime = now;
          }
          
          // Throttle excessive logging if getting too many messages
          const shouldLog = messageCount < 20;
          
          // Only try to parse if there's actual data
          if (event.data && event.data.trim() !== '') {
            const data = JSON.parse(event.data);
            
            // Handle recommendations from Lambda
            if (data.recommendations) {
              // Always log recommendations
              console.log('Received game recommendations:', data.recommendations);
              if (this.callbacks.onRecommendations) {
                this.callbacks.onRecommendations(data);
              }
            }
            // Handle game options response (making this the first check to prioritize it)
            else if (data.games && Array.isArray(data.games)) {
              console.log(`Received game options: ${data.games.length} games`);
              
              // Check if we have a valid and reasonable number of games
              if (data.games.length > 0 && data.games.every(game => typeof game === 'string')) {
                if (this.callbacks.onGameOptions) {
                  try {
                    // Make an explicit copy of the callback before calling it
                    const callback = this.callbacks.onGameOptions;
                    // Clear it immediately to prevent double-handling
                    this.callbacks.onGameOptions = null; 
                    // Call the callback with the data
                    callback(data);
                  } catch (callbackError) {
                    console.error('Error in game options callback:', callbackError);
                  }
                } else {
                  console.log('Received game options but no callback registered');
                }
              } else {
                console.warn('Received invalid game options data');
              }
            }
            // Handle errors from the Lambda function, but also forward to game options callback if relevant
            else if (data.message && (data.message.includes("error") || data.success === false)) {
              // Reduce error logging if getting too many
              if (shouldLog) {
                console.warn('Server reported an error:', data.message);
              }
              
              // Check if this might be a response to a game options request
              if (this.callbacks.onGameOptions && 
                  (data.action === 'getGameOptions' || data.requestId)) {
                // Forward the error to the game options callback
                try {
                  const callback = this.callbacks.onGameOptions;
                  this.callbacks.onGameOptions = null;
                  callback(data);
                } catch (callbackError) {
                  console.error('Error in game options error callback:', callbackError);
                }
              }
              
              // If getting too many errors, consider reconnecting
              if (messageCount > 10 && data.message.includes("error")) {
                console.warn('Too many errors, consider reconnecting');
              }
            }
            // Log other messages only if not excessive
            else if (shouldLog) {
              console.log('Received WebSocket message:', data);
            }
          } else if (shouldLog) {
            // Only log empty messages if not getting too many
            console.log('Received empty WebSocket message');
          }
        } catch (error) {
          if (messageCount < 5) {  // Only log a few parse errors
            console.error('Error parsing WebSocket message:', error);
          }
        }
      };
    });
  }

  disconnect() {
    if (this.socket) {
      this.socket.close();
      this.socket = null;
      this.isConnected = false;
    }
  }

  sendUserData(userData) {
    // Format message to match Lambda's expected structure
    // Make sure action is specifically sendUserData as expected by Lambda
    const action = userData.action || 'sendUserData';
    
    // Create a clean message object with required fields
    const message = {
      action: action,
      username: userData.username || this.username || 'anonymous',
      // Removed age since we no longer use it
      modelType: userData.modelType, // Don't use a default - we need to preserve exactly what's selected
      games: userData.games || userData.hobbies || []
    };
    
    // Only use Traditional as a fallback if modelType is completely undefined or null
    if (message.modelType === undefined || message.modelType === null || message.modelType === '') {
      console.warn('No modelType specified in user data, defaulting to Traditional');
      message.modelType = 'Traditional';
    }
    
    // Log the model type being sent to ensure it's correct
    console.log(`Using model type: ${message.modelType}`);
    
    // Add game ratings if available
    if (userData.gameRatings) {
      message.gameRatings = userData.gameRatings;
    }
    
    // Add any additional fields that are safe to include
    // Avoid spreading the entire userData to prevent duplicate/conflicting fields
    if (userData.preferences) message.preferences = userData.preferences;
    if (userData.timestamp) message.timestamp = userData.timestamp;
    
    console.log('Sending user data to WebSocket:', message);
    
    // Ensure the connection is active before sending
    if (this.isConnected) {
      this._sendMessage(message);
    } else {
      console.warn('WebSocket not connected, attempting to queue message');
      this.messageQueue.push(message);
      
      // If we have a username, try to reconnect
      if (this.username) {
        this.connect(this.username)
          .then(() => console.log('Reconnected to WebSocket'))
          .catch(err => console.error('Failed to reconnect:', err));
      }
    }
  }
  
  // Get default games to use as fallback
  getDefaultGames() {
    return [
      'Minecraft', 'Fortnite', 'Zelda', 'Mario',
      'Pokemon', 'GTA', 'Call of Duty', 'FIFA',
      'Skyrim'
    ];
  }
  
  // Simplified function that just returns the default games without making network requests
  requestGameOptions(count = 9) {
    return new Promise((resolve) => {
      // Get default games
      const games = this.getDefaultGames();
      
      // Simulate network delay
      setTimeout(() => {
        console.log('Using local game options without network request');
        resolve(games);
      }, 100);
    });
  }
  
  onRecommendations(callback) {
    this.callbacks.onRecommendations = callback;
  }
  
  _sendMessage(message) {
    // Add a timestamp to the message to identify when it was created
    const messageWithTimestamp = {
      ...message,
      _timestamp: Date.now(),
      _retryCount: message._retryCount || 0
    };
    
    // Only retry messages for up to 30 seconds
    const isTooOld = messageWithTimestamp._timestamp < (Date.now() - 30000);
    const tooManyRetries = messageWithTimestamp._retryCount > 3;
    
    // Skip messages that have been retried too many times or are too old
    if (isTooOld || tooManyRetries) {
      console.warn(`Dropping message: ${isTooOld ? 'too old' : 'too many retries'}`);
      return;
    }
    
    if (this.isConnected && this.socket && this.socket.readyState === WebSocket.OPEN) {
      try {
        // Clone the message and remove internal properties
        const cleanMessage = { ...messageWithTimestamp };
        delete cleanMessage._timestamp;
        delete cleanMessage._retryCount;
        
        // Format message as JSON and send
        const messageStr = JSON.stringify(cleanMessage);
        this.socket.send(messageStr);
        
        // Only log successful sends if not retrying too frequently
        if (messageWithTimestamp._retryCount === 0) {
          console.log('Message sent successfully');
        }
      } catch (error) {
        console.error('Error sending message:', error);
        
        // Increment retry count and queue
        messageWithTimestamp._retryCount += 1;
        this.messageQueue.push(messageWithTimestamp);
      }
    } else {
      // Only log once when initially queuing
      if (messageWithTimestamp._retryCount === 0) {
        console.warn('WebSocket not ready, queuing message');
      }
      
      // Queue the message to send when connected
      this.messageQueue.push(messageWithTimestamp);
    }
  }
  
  _processQueue() {
    const queueLength = this.messageQueue.length;
    if (queueLength === 0) {
      return; // Nothing to process
    }
    
    // Only log if there are actually messages to process
    console.log(`Processing message queue (${queueLength} items)...`);
    
    // Filter out any stale messages before processing
    const now = Date.now();
    const validMessages = this.messageQueue.filter(msg => {
      const isTooOld = msg._timestamp && msg._timestamp < (now - 30000);
      const tooManyRetries = msg._retryCount && msg._retryCount > 3;
      return !isTooOld && !tooManyRetries;
    });
    
    // Replace the queue with filtered messages
    this.messageQueue = validMessages;
    
    // If the WebSocket isn't connected, don't try to send
    if (!this.isConnected || !this.socket || this.socket.readyState !== WebSocket.OPEN) {
      console.warn('WebSocket not connected, keeping messages in queue for later');
      return;
    }
    
    // Process each message with built-in rate limiting
    let messagesSent = 0;
    const processMessages = () => {
      // Stop if we've processed all messages or connection is gone
      if (this.messageQueue.length === 0 || !this.isConnected) {
        return;
      }
      
      // Get the next message
      const message = this.messageQueue.shift();
      
      // Send using the main send function (which handles retries)
      this._sendMessage(message);
      messagesSent++;
      
      // Process more messages with delay if there are any left
      if (this.messageQueue.length > 0) {
        // Use increasing delays as we send more messages to avoid flooding
        const delay = Math.min(100 * messagesSent, 1000);
        setTimeout(processMessages, delay);
      }
    };
    
    // Start processing
    processMessages();
  }
}

export default new WebSocketService();