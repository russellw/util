const readline = require('readline');

// Create the readline interface
const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout
});

// Game state
const gameState = {
  playerName: '',
  currentLocation: 'start',
  inventory: [],
  hasKey: false,
  gameOver: false
};

// Game map/locations
const locations = {
  start: {
    description: "You find yourself at the entrance of a mysterious cave. The air is damp and cold. Paths lead to the west and to the north.",
    exits: {
      west: 'darkCorridor',
      north: 'glowingChamber'
    }
  },
  darkCorridor: {
    description: "You're in a dark corridor. Water drips from the ceiling. There's a faint scraping sound in the distance.",
    exits: {
      east: 'start',
      north: 'treasureRoom'
    }
  },
  glowingChamber: {
    description: "This chamber is illuminated by strange glowing crystals. In the center is a pedestal with something shiny.",
    exits: {
      south: 'start',
      east: 'puzzleRoom'
    },
    items: ['key']
  },
  treasureRoom: {
    description: "You've found the treasure room! But the door is locked and requires a key.",
    exits: {
      south: 'darkCorridor'
    },
    locked: true,
    lockedDescription: "The door is locked. You need a key to enter.",
    unlockMessage: "You use the key to unlock the door and enter the treasure room."
  },
  puzzleRoom: {
    description: "This room contains an ancient puzzle. Solving it might reveal something valuable.",
    exits: {
      west: 'glowingChamber'
    },
    puzzle: {
      solved: false,
      question: "What has keys but can't open locks?",
      answer: "piano",
      successMessage: "The wall slides open, revealing a secret passage!",
      revealExit: {
        direction: "north",
        destination: "secretRoom"
      }
    }
  },
  secretRoom: {
    description: "You've discovered a secret room filled with ancient artifacts and mysterious writings.",
    exits: {
      south: 'puzzleRoom'
    }
  }
};

// Game commands
const commands = {
  help: () => {
    console.log("\nAvailable commands:");
    console.log("look - Look around the current location");
    console.log("go [direction] - Move in a direction (north, south, east, west)");
    console.log("take [item] - Pick up an item");
    console.log("inventory - Check your inventory");
    console.log("use [item] - Use an item");
    console.log("solve [answer] - Solve a puzzle");
    console.log("quit - Exit the game");
    promptUser();
  },
  
  look: () => {
    displayLocation();
    promptUser();
  },
  
  go: (direction) => {
    const currentLocation = locations[gameState.currentLocation];
    
    if (!direction) {
      console.log("Go where? Please specify a direction.");
      return promptUser();
    }
    
    if (currentLocation.exits[direction]) {
      const destinationId = currentLocation.exits[direction];
      const destination = locations[destinationId];
      
      // Check if the destination is locked
      if (destination.locked && destination.locked === true) {
        console.log(destination.lockedDescription);
        return promptUser();
      }
      
      gameState.currentLocation = destinationId;
      displayLocation();
    } else {
      console.log("You can't go that way.");
    }
    promptUser();
  },
  
  take: (item) => {
    if (!item) {
      console.log("Take what?");
      return promptUser();
    }
    
    const currentLocation = locations[gameState.currentLocation];
    
    if (currentLocation.items && currentLocation.items.includes(item)) {
      // Remove the item from the location
      const itemIndex = currentLocation.items.indexOf(item);
      currentLocation.items.splice(itemIndex, 1);
      
      // Add to inventory
      gameState.inventory.push(item);
      
      // Special handling for key
      if (item === 'key') {
        gameState.hasKey = true;
        console.log("You picked up a key. It might be useful somewhere.");
      } else {
        console.log(`You picked up the ${item}.`);
      }
    } else {
      console.log(`There's no ${item} here.`);
    }
    promptUser();
  },
  
  inventory: () => {
    if (gameState.inventory.length === 0) {
      console.log("Your inventory is empty.");
    } else {
      console.log("Inventory:");
      gameState.inventory.forEach(item => console.log(`- ${item}`));
    }
    promptUser();
  },
  
  use: (item) => {
    if (!item) {
      console.log("Use what?");
      return promptUser();
    }
    
    if (!gameState.inventory.includes(item)) {
      console.log(`You don't have a ${item} to use.`);
      return promptUser();
    }
    
    // Using specific items
    if (item === 'key' && gameState.currentLocation === 'treasureRoom') {
      console.log(locations.treasureRoom.unlockMessage);
      locations.treasureRoom.locked = false;
      
      // Customize the win condition
      console.log("\n🎉 Congratulations! You've found the ancient treasure!");
      console.log("Mountains of gold and jewels surround you. Your adventure has been successful!");
      gameState.gameOver = true;
    } else {
      console.log(`You can't use the ${item} here.`);
    }
    promptUser();
  },
  
  solve: (answer) => {
    const currentLocation = locations[gameState.currentLocation];
    
    if (!currentLocation.puzzle) {
      console.log("There's no puzzle to solve here.");
      return promptUser();
    }
    
    if (currentLocation.puzzle.solved) {
      console.log("You've already solved this puzzle.");
      return promptUser();
    }
    
    if (!answer) {
      console.log(`The puzzle asks: "${currentLocation.puzzle.question}"`);
      return promptUser();
    }
    
    if (answer.toLowerCase() === currentLocation.puzzle.answer.toLowerCase()) {
      console.log(currentLocation.puzzle.successMessage);
      currentLocation.puzzle.solved = true;
      
      // Add new exit if puzzle reveals one
      if (currentLocation.puzzle.revealExit) {
        const { direction, destination } = currentLocation.puzzle.revealExit;
        currentLocation.exits[direction] = destination;
      }
    } else {
      console.log("That's not the correct answer. Try again.");
    }
    promptUser();
  },
  
  quit: () => {
    console.log("Thanks for playing! Goodbye.");
    rl.close();
    process.exit(0);
  }
};

// Display the current location
function displayLocation() {
  const location = locations[gameState.currentLocation];
  console.log(`\n${location.description}`);
  
  // List available exits
  const exits = Object.keys(location.exits);
  if (exits.length > 0) {
    console.log("Exits: " + exits.join(", "));
  }
  
  // List available items
  if (location.items && location.items.length > 0) {
    console.log("You can see: " + location.items.join(", "));
  }
  
  // Display puzzle if not solved
  if (location.puzzle && !location.puzzle.solved) {
    console.log(`There's a puzzle here: "${location.puzzle.question}"`);
  }
}

// Process user commands
function processCommand(input) {
  const tokens = input.toLowerCase().trim().split(' ');
  const command = tokens[0];
  const arg = tokens.slice(1).join(' ');
  
  if (commands[command]) {
    commands[command](arg);
  } else {
    console.log("I don't understand that command. Type 'help' for a list of commands.");
    promptUser();
  }
}

// Prompt for user input
function promptUser() {
  if (gameState.gameOver) {
    console.log("\nGame over! Type 'restart' to play again or 'quit' to exit.");
  }
  
  rl.question('> ', (input) => {
    if (input.toLowerCase() === 'restart') {
      startGame();
    } else {
      processCommand(input);
    }
  });
}

// Start the game
function startGame() {
  // Reset game state
  gameState.currentLocation = 'start';
  gameState.inventory = [];
  gameState.hasKey = false;
  gameState.gameOver = false;
  
  // Introduction
  console.log("\n==================================");
  console.log("        THE CAVE OF WONDERS       ");
  console.log("==================================\n");
  
  rl.question("What is your name, adventurer? ", (name) => {
    gameState.playerName = name;
    console.log(`\nWelcome, ${name}! Your adventure begins now. Type 'help' for commands.\n`);
    displayLocation();
    promptUser();
  });
}

// Initialize the game
startGame();
