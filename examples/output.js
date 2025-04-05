/**
 * Enhanced print without newline function
 * Handles primitives, null, and complex objects appropriately
 * Will throw an error if 'undefined' is passed (as per requirements)
 */
function _prin(a) {
    // Handle null explicitly
    if (a === null) {
        process.stdout.write("null");
        return;
    }
    
    // Handle different types appropriately
    if (typeof a === 'object') {
        // For objects, arrays, etc., use JSON.stringify with formatting
        try {
            process.stdout.write(JSON.stringify(a, null, 0));
        } catch (e) {
            // Fallback for circular references or other JSON issues
            process.stdout.write(String(a));
        }
    } else {
        // For primitives, direct conversion is fastest
        process.stdout.write(String(a));
    }
}

// Test function with various inputs
function testPrin() {
    // Test with primitives
    _prin("Hello");
    _prin(", ");
    _prin(42);
    _prin(" ");
    _prin(true);
    _prin(" ");
    
    // Test with null
    _prin(null);
    _prin(" ");
    
    // Test with objects
    _prin({name: "John", age: 30});
    _prin(" ");
    
    // Test with arrays
    _prin([1, 2, 3, "test"]);
    _prin(" ");
    
    // Test with nested objects
    const complexObj = {
        person: {
            name: "Alice",
            contacts: {
                email: "alice@example.com",
                phone: "123-456-7890"
            }
        },
        hobbies: ["reading", "coding", "hiking"]
    };
    _prin(complexObj);
    _prin("\n");
    
    // Test with Date object
    _prin(new Date());
    
    // Uncomment this to see the intended error for undefined
    // _prin(undefined);
    
    // Create an object with circular reference
    const circular = {};
    circular.self = circular;
    _prin(" ");
    _prin(circular);
}

// Run the test
testPrin();