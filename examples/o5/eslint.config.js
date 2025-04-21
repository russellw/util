import js from "@eslint/js"
import globals from "globals"
import { includeIgnoreFile } from "@eslint/compat"
import { fileURLToPath } from "node:url"

const gitignorePath = fileURLToPath(new URL(".gitignore", import.meta.url))

export default [
	// Include gitignore patterns first
	includeIgnoreFile(gitignorePath),

	// Your existing configuration
	js.configs.recommended,
	{
		files: ["**/*.js"],
		languageOptions: {
			ecmaVersion: 2023,
			sourceType: "module",
			globals: {
				...globals.browser,
				...globals.node,
				myCustomGlobal: "readonly",
			},
		},
		rules: {
			curly: ["error", "all"],
			// Other rules...
		},
	},
	// Any other configuration objects you had before
]
