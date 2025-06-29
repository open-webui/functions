# 🎮 Games Hub Filter

A streamlined OpenWebUI filter that automatically installs and manages a retro games hub with auto-discovery and configuration.

## ✨ Features

🚀 **Auto-Installation** - Downloads and sets up Games Hub on first use  
🔍 **Smart Game Scanner** - Automatically discovers games in your directory  
⚙️ **Auto-Config Generation** - Creates fresh `games_config.json` with found games  
🎨 **Theme Support** - Dark, light, and ultra-dark themes  
🪟 **Custom Game Windows** - Optimized window sizes per game/category  
🛡️ **Secure Downloads** - Domain validation and content verification  
⌨️ **Keyboard Shortcuts** - Full keyboard navigation support  

## 📦 Installation

1. **Add the filter** to your OpenWebUI functions directory
2. **Enable the filter** in your OpenWebUI settings
3. **Run ```!games``` or ```!games scan```  command** - that's it! ✅

## 🎯 Usage

### 🎮 Launch Games Hub
```
!games
```
- Downloads and installs Games Hub (first time only)
- Scans games directory and generates config
- Opens the Games Hub in a new window

### 🔄 Rescan Games Directory  
```
!games scan
```
- Rescans the games directory
- Regenerates `games_config.json` with current games
- Use when you add/remove games

## 📁 File Structure

```
/cache/functions/games_hub/
├── 📄 index.html              # Main hub interface
├── 🎨 style.css               # Themes and styling
├── ⚙️ main.js                 # Core application
├── 🛠️ utils.js                # Utility functions
├── 🎮 game-manager.js         # Game launching
├── 🎨 theme-manager.js        # Theme switching  
├── 🔧 config-manager.js       # Config loading
├── 🎛️ ui-manager.js           # Interface management
├── ⌨️ event-manager.js        # Event handling
├── 📂 config/
│   └── 🎯 games_config.json   # Auto-generated game list
└── 🎮 games/
    ├── 🐍 snake-game/
    │   └── index.html
    ├── 🏓 pong/
    │   └── index.html
    └── 🧩 tetris/
        └── index.html
```

## 🎮 Adding Games

1. **Create game directory**: `/cache/functions/games_hub/games/[game-name]/`
2. **Add your game**: Place `index.html` and assets in the directory
3. **Rescan**: Run `!games scan` to detect the new game
4. **Play**: Game appears in the hub automatically! 🎉

### 📋 Supported Game Formats
- 📁 **Directory-based**: `games/my-game/index.html`
- 📄 **Single files**: `games/my-game.html`
- 🎯 **Auto-detection**: Finds `index.html`, `game.html`, or `[game-name].html`

## ⚙️ Configuration

### 🔧 Filter Settings (Valves)

| Setting | Default | Description |
|---------|---------|-------------|
| 🚀 `install_on_startup` | `true` | Auto-install on first use |
| 🔄 `force_reinstall` | `false` | Force fresh installation |
| 🌐 `hub_repo_url` | GitHub URL | Source repository |
| 🛡️ `trusted_domains` | `github.com` | Allowed download domains |
| ⏱️ `download_timeout` | `30` | Download timeout (seconds) |
| 💬 `show_status_messages` | `true` | Show installation progress |
| 🔍 `auto_scan_games` | `true` | Auto-scan after install |
| 🐛 `debug_mode` | `false` | Enable debug logging |

### 🎨 Theme Options
- 🌙 **Dark** (default)
- ☀️ **Light** 
- 🌑 **Ultra Dark**

## 🚨 Troubleshooting

### ❌ Games Not Launching
- Check that games have an `index.html` file
- Run `!games scan` to refresh the game list
- Verify file paths in `/cache/functions/games_hub/config/games_config.json`

### 🔄 Installation Issues  
- Enable `debug_mode` in filter settings
- Check server logs for detailed error messages
- Verify `trusted_domains` includes your download source

### 🎮 Games Not Detected
- Ensure games are in `/cache/functions/games_hub/games/[game-name]/`
- Each game needs an `index.html` file
- Run `!games scan` after adding games

## 📊 Version Info

**Current Version**: 8.5.0  
**Architecture**: Modular Python + JavaScript  
**Compatibility**: OpenWebUI Functions  
**License**: Open Source  

## 🔗 Quick Links

- 🎮 **Add Games**: Place HTML5 games in the games directory
- 🔄 **Rescan**: `!games scan` 
- 🎨 **Themes**: Toggle with button or `Ctrl+T`
- ⌨️ **Shortcuts**: `Ctrl+Shift+H` for help

---

**🎉 Ready to play? Run `!games` and start your retro gaming adventure!**