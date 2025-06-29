"""
title: Games Hub Filter
author: pkeffect
author_url: https://github.com/pkeffect
funding_url: https://github.com/open-webui
version: 8.5.1
description: Environment-agnostic Games Hub with robust auto-installation from GitHub. Auto-scans games directory and generates fresh config. Commands: !games (launch), !games scan (regenerate config).
"""

from pydantic import BaseModel, Field
from typing import Optional, Dict, List, Callable, Any
import time
import json
import asyncio
import aiofiles
import os
import shutil
import tempfile
import urllib.request
import urllib.parse
import urllib.error
import zipfile
import traceback
from pathlib import Path
import logging

# --- Module-Level State Management ---
_INIT_LOCK = asyncio.Lock()
_HUB_READY = False
_IS_INITIALIZING = False

# Use a proper logger to ensure output appears in server logs
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def get_plugin_cache_dir(plugin_name: str) -> Path:
    """
    Determines the absolute path to the plugin's cache directory in an environment-agnostic way.
    Works both in Docker containers and native installations with multiple fallback strategies.

    Args:
        plugin_name: The name of the plugin (e.g., 'games_hub')

    Returns:
        A pathlib.Path object representing the absolute path to the cache directory.

    Raises:
        FileNotFoundError: If the project root cannot be determined.
    """

    def try_path_discovery():
        # Start with the absolute path of the current script file
        current_file_path = Path(__file__).resolve()
        logger.info(
            f"[Cache Dir Discovery] Starting from script location: {current_file_path}"
        )

        # Strategy 1: Look for 'backend' directory (standard Open WebUI structure)
        project_root = current_file_path
        checked_paths = []

        while project_root != project_root.parent:  # While not at filesystem root
            checked_paths.append(str(project_root))
            logger.info(f"[Cache Dir Discovery] Checking directory: {project_root}")

            if (project_root / "backend").is_dir():
                logger.info(
                    f"[Cache Dir Discovery] Found 'backend' directory at: {project_root}"
                )
                cache_dir = (
                    project_root
                    / "backend"
                    / "data"
                    / "cache"
                    / "functions"
                    / plugin_name
                )
                cache_dir.mkdir(parents=True, exist_ok=True)
                return cache_dir

            project_root = project_root.parent

        logger.warning(
            f"[Cache Dir Discovery] Strategy 1 failed. Checked paths: {checked_paths}"
        )

        # Strategy 2: Look for other Open WebUI markers
        project_root = current_file_path
        markers = [
            "frontend",
            "src",
            "package.json",
            "requirements.txt",
            "docker-compose.yml",
        ]

        while project_root != project_root.parent:
            for marker in markers:
                if (project_root / marker).exists():
                    logger.info(
                        f"[Cache Dir Discovery] Found Open WebUI marker '{marker}' at: {project_root}"
                    )
                    # Try different cache directory structures
                    possible_cache_paths = [
                        project_root
                        / "backend"
                        / "data"
                        / "cache"
                        / "functions"
                        / plugin_name,
                        project_root / "data" / "cache" / "functions" / plugin_name,
                        project_root / "cache" / "functions" / plugin_name,
                        project_root / "backend" / "cache" / "functions" / plugin_name,
                    ]

                    for cache_path in possible_cache_paths:
                        try:
                            cache_path.mkdir(parents=True, exist_ok=True)
                            logger.info(
                                f"[Cache Dir Discovery] Successfully created cache directory: {cache_path}"
                            )
                            return cache_path
                        except Exception as e:
                            logger.warning(
                                f"[Cache Dir Discovery] Failed to create {cache_path}: {e}"
                            )
                            continue

            project_root = project_root.parent

        # Strategy 3: Environment variable fallback
        if "OPEN_WEBUI_DATA_DIR" in os.environ:
            data_dir = Path(os.environ["OPEN_WEBUI_DATA_DIR"])
            cache_dir = data_dir / "cache" / "functions" / plugin_name
            logger.info(f"[Cache Dir Discovery] Using OPEN_WEBUI_DATA_DIR: {cache_dir}")
            cache_dir.mkdir(parents=True, exist_ok=True)
            return cache_dir

        # Strategy 4: Current working directory fallback
        cwd = Path.cwd()
        if "open-webui" in str(cwd).lower() or "openwebui" in str(cwd).lower():
            cache_dir = cwd / "data" / "cache" / "functions" / plugin_name
            logger.info(
                f"[Cache Dir Discovery] Using current working directory fallback: {cache_dir}"
            )
            cache_dir.mkdir(parents=True, exist_ok=True)
            return cache_dir

        return None

    try:
        cache_dir = try_path_discovery()
        if cache_dir:
            logger.info(
                f"[Cache Dir] Successfully determined cache directory: {cache_dir}"
            )
            return cache_dir
        else:
            raise FileNotFoundError("All discovery strategies failed")

    except Exception as e:
        # Final fallback: Use a temporary directory with warning
        temp_cache = Path.home() / ".cache" / "open-webui-games-hub" / plugin_name
        logger.error(
            f"[Cache Dir] Failed to find proper cache directory, using fallback: {temp_cache}"
        )
        logger.error(f"[Cache Dir] Original error: {e}")
        logger.error(f"[Cache Dir] Script location: {Path(__file__).resolve()}")
        logger.error(f"[Cache Dir] Current working directory: {Path.cwd()}")
        logger.error(f"[Cache Dir] Environment variables: {dict(os.environ)}")

        try:
            temp_cache.mkdir(parents=True, exist_ok=True)
            logger.warning(
                f"[Cache Dir] Using temporary fallback directory: {temp_cache}"
            )
            return temp_cache
        except Exception as fallback_error:
            raise FileNotFoundError(
                f"Could not determine plugin cache directory and fallback failed. "
                f"Original error: {e}. Fallback error: {fallback_error}. "
                f"Script location: {Path(__file__).resolve()}. "
                f"Please check your Open WebUI installation."
            )


class HubDownloadManager:
    """Advanced download manager for Games Hub files with robust error handling."""

    def __init__(self, valves):
        self.valves = valves
        # Use custom cache path if provided, otherwise use environment-agnostic discovery
        if valves.custom_cache_path.strip():
            custom_path = Path(valves.custom_cache_path.strip()).expanduser().resolve()
            self.hub_root_path = custom_path / "games_hub"
            logger.info(f"[Cache Dir] Using custom cache path: {self.hub_root_path}")
            # Ensure the directory exists
            self.hub_root_path.mkdir(parents=True, exist_ok=True)
        else:
            self.hub_root_path = get_plugin_cache_dir("games_hub")

        self.hub_index_path = self.hub_root_path / "index.html"

    def check_installation(self) -> bool:
        """Check if Games Hub is properly installed."""
        if not self.hub_index_path.is_file():
            logger.info(
                f"[Games Hub Check] index.html not found at: {self.hub_index_path}"
            )
            return False

        # Check for essential Games Hub files based on actual structure
        essential_files = [
            "index.html",
            "style.css",
            "main.js",
            "utils.js",
            "game-manager.js",
            "event-manager.js",
            "theme-manager.js",
            "ui-manager.js",
            "config-manager.js",  # This is in root directory, not config/
        ]

        essential_dirs = ["config", "games"]

        missing_files = []
        missing_dirs = []

        # Check essential files
        for file in essential_files:
            file_path = self.hub_root_path / file
            if not file_path.exists():
                missing_files.append(file)

        # Check essential directories
        for dir_name in essential_dirs:
            dir_path = self.hub_root_path / dir_name
            if not dir_path.exists() or not dir_path.is_dir():
                missing_dirs.append(dir_name)

        # Check for games_config.json inside config directory (optional - we generate this)
        games_config_path = self.hub_root_path / "config" / "games_config.json"
        if not games_config_path.exists():
            logger.info(
                "[Games Hub Check] games_config.json not found - will be generated during scan"
            )

        if missing_files or missing_dirs:
            logger.warning(
                f"[Games Hub Check] Missing files: {missing_files}, Missing dirs: {missing_dirs}"
            )
            return False

        logger.info(
            f"[Games Hub Check] Installation verified - all essential files and directories present"
        )
        return True

    def is_trusted_domain(self, url: str) -> bool:
        """Check if URL domain is in the trusted whitelist."""
        try:
            logger.info(f"[Domain Check] Validating URL: {url}")
            parsed = urllib.parse.urlparse(url)
            logger.info(
                f"[Domain Check] Parsed - scheme: {parsed.scheme}, netloc: {parsed.netloc}"
            )

            if not parsed.scheme or parsed.scheme.lower() not in ["https"]:
                logger.error(f"[Domain Check] Invalid scheme: '{parsed.scheme}'")
                return False

            trusted_domains_raw = self.valves.trusted_domains
            trusted_domains = [
                d.strip().lower() for d in trusted_domains_raw.split(",")
            ]
            logger.info(f"[Domain Check] Trusted domains: {trusted_domains}")
            logger.info(f"[Domain Check] URL netloc: '{parsed.netloc.lower()}'")

            is_trusted = parsed.netloc.lower() in trusted_domains
            logger.info(f"[Domain Check] Result: {is_trusted}")
            return is_trusted

        except Exception as e:
            logger.error(f"[Domain Check] Exception: {e}")
            traceback.print_exc()
            return False

    def validate_download_content(self, content_path: Path) -> Dict:
        """Validate that downloaded content contains expected Games Hub files."""
        validation_result = {
            "valid": False,
            "errors": [],
            "found_files": [],
            "missing_files": [],
            "missing_dirs": [],
        }

        try:
            if not content_path.exists() or not content_path.is_dir():
                validation_result["errors"].append(
                    "Download path does not exist or is not a directory"
                )
                return validation_result

            # Check for essential Games Hub files based on actual structure
            essential_files = [
                "index.html",
                "style.css",
                "main.js",
                "utils.js",
                "game-manager.js",
                "event-manager.js",
                "theme-manager.js",
                "ui-manager.js",
                "config-manager.js",  # This is in root directory, not config/
            ]

            essential_dirs = ["config", "games"]
            # Note: games_config.json is generated during scan, not required in download

            all_files = []
            all_dirs = []

            # Catalog all downloaded content
            for item in content_path.rglob("*"):
                relative_path = item.relative_to(content_path)
                if item.is_file():
                    all_files.append(str(relative_path))
                elif item.is_dir():
                    all_dirs.append(str(relative_path))

            validation_result["found_files"] = all_files
            logger.info(
                f"[Validation] Found {len(all_files)} files and {len(all_dirs)} directories in download"
            )

            # Check essential files
            missing_essential = []
            for essential_file in essential_files:
                found = any(
                    file_path == essential_file
                    or file_path.endswith(f"/{essential_file}")
                    for file_path in all_files
                )
                if not found:
                    missing_essential.append(essential_file)

            # Check essential directories
            missing_dirs = []
            for essential_dir in essential_dirs:
                found = any(
                    dir_path == essential_dir
                    or dir_path.endswith(f"/{essential_dir}")
                    or dir_path.startswith(f"{essential_dir}/")
                    for dir_path in all_dirs
                )
                if not found:
                    missing_dirs.append(essential_dir)

            # Check essential nested files - none required since we generate games_config.json

            validation_result["missing_files"] = missing_essential
            validation_result["missing_dirs"] = missing_dirs

            if missing_essential:
                validation_result["errors"].append(
                    f"Missing essential files: {missing_essential}"
                )

            if missing_dirs:
                validation_result["errors"].append(
                    f"Missing essential directories: {missing_dirs}"
                )

            if missing_essential or missing_dirs:
                logger.error(
                    f"[Validation] Missing files: {missing_essential}, Missing dirs: {missing_dirs}"
                )
                return validation_result

            # Validate index.html contains Games Hub content
            index_files = [f for f in all_files if f.endswith("index.html")]
            if index_files:
                # Try to find the main index.html (not in subdirectories)
                main_index = next(
                    (f for f in index_files if "/" not in f), index_files[0]
                )
                index_path = content_path / main_index

                try:
                    with open(index_path, "r", encoding="utf-8") as f:
                        index_content = f.read()

                    # Look for Games Hub indicators
                    hub_indicators = ["games", "hub", "main.js", "game-manager"]
                    if not any(
                        indicator.lower() in index_content.lower()
                        for indicator in hub_indicators
                    ):
                        validation_result["errors"].append(
                            "index.html does not appear to contain Games Hub content"
                        )
                        return validation_result

                except Exception as e:
                    validation_result["errors"].append(
                        f"Could not read index.html: {e}"
                    )
                    return validation_result

            # Check that games directory has content
            games_files = [
                f for f in all_files if f.startswith("games/") or "/games/" in f
            ]
            if not games_files:
                logger.warning(
                    "[Validation] No files found in games directory - this might be expected if games are loaded dynamically"
                )
            else:
                logger.info(
                    f"[Validation] Found {len(games_files)} files in games directory"
                )

            validation_result["valid"] = True
            logger.info(
                f"[Validation] ✅ Content validation passed - all essential Games Hub files found"
            )
            return validation_result

        except Exception as e:
            validation_result["errors"].append(f"Validation exception: {e}")
            logger.error(f"[Validation] Exception during validation: {e}")
            traceback.print_exc()
            return validation_result

    async def download_and_extract_hub(self) -> Dict:
        """Download and extract Games Hub with comprehensive error handling."""
        repo_url = self.valves.hub_repo_url
        logger.info(f"[Games Hub Download] Starting installation from: {repo_url}")

        result = {
            "success": False,
            "error": None,
            "url": repo_url,
            "files_extracted": 0,
            "validation_result": None,
        }

        # Validate domain
        if not self.is_trusted_domain(repo_url):
            result["error"] = (
                f"Untrusted domain. Allowed domains: {self.valves.trusted_domains}"
            )
            logger.error(f"[Games Hub Download] {result['error']}")
            return result

        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                zip_path = temp_path / "hub.zip"
                extract_path = temp_path / "extracted"

                # Download with timeout and proper headers
                logger.info("[Games Hub Download] Downloading ZIP archive...")
                req = urllib.request.Request(
                    repo_url,
                    headers={
                        "User-Agent": "OpenWebUI-GamesHub-Filter/8.5.1",
                        "Accept": "application/zip, application/octet-stream, */*",
                    },
                )

                with urllib.request.urlopen(
                    req, timeout=self.valves.download_timeout
                ) as response:
                    if response.status != 200:
                        result["error"] = f"HTTP {response.status}: {response.reason}"
                        logger.error(f"[Games Hub Download] {result['error']}")
                        return result

                    # Check content type
                    content_type = response.headers.get("Content-Type", "")
                    logger.info(f"[Games Hub Download] Content-Type: {content_type}")

                    # Read content with size limit
                    content_length = response.headers.get("Content-Length")
                    if content_length:
                        size = int(content_length)
                        if size > 50 * 1024 * 1024:  # 50MB limit
                            result["error"] = (
                                f"Download too large: {size} bytes (max 50MB)"
                            )
                            logger.error(f"[Games Hub Download] {result['error']}")
                            return result
                        logger.info(
                            f"[Games Hub Download] Expected size: {size:,} bytes"
                        )

                    with open(zip_path, "wb") as out_file:
                        shutil.copyfileobj(response, out_file)

                logger.info(
                    f"[Games Hub Download] Download complete: {zip_path.stat().st_size:,} bytes"
                )

                # Extract with validation
                logger.info("[Games Hub Download] Extracting files...")
                try:
                    with zipfile.ZipFile(zip_path, "r") as zf:
                        # Validate ZIP file
                        bad_file = zf.testzip()
                        if bad_file:
                            result["error"] = f"Corrupted ZIP file: {bad_file}"
                            logger.error(f"[Games Hub Download] {result['error']}")
                            return result

                        # Extract all files
                        zf.extractall(extract_path)
                        result["files_extracted"] = len(zf.namelist())
                        logger.info(
                            f"[Games Hub Download] Extracted {result['files_extracted']} files"
                        )

                        # Debug: list extracted files if in debug mode
                        if self.valves.debug_mode:
                            logger.info(
                                f"[Games Hub Download] Extracted files: {zf.namelist()[:10]}{'...' if len(zf.namelist()) > 10 else ''}"
                            )

                except zipfile.BadZipFile as e:
                    result["error"] = f"Invalid ZIP file: {e}"
                    logger.error(f"[Games Hub Download] {result['error']}")
                    return result

                # Find source directory (GitHub repos extract to a subdirectory)
                extracted_items = list(extract_path.iterdir())
                if not extracted_items:
                    result["error"] = "ZIP file appears to be empty"
                    logger.error(f"[Games Hub Download] {result['error']}")
                    return result

                # Find the main content directory
                source_dir = None
                if len(extracted_items) == 1 and extracted_items[0].is_dir():
                    # Single directory (typical GitHub archive structure)
                    source_dir = extracted_items[0]
                else:
                    # Multiple items at root level
                    source_dir = extract_path

                logger.info(
                    f"[Games Hub Download] Using source directory: {source_dir}"
                )

                # Debug: show directory contents if in debug mode
                if self.valves.debug_mode:
                    items = list(source_dir.iterdir())
                    logger.info(
                        f"[Games Hub Download] Source directory contains: {[item.name for item in items[:10]]}"
                    )
                    if len(items) > 10:
                        logger.info(
                            f"[Games Hub Download] ... and {len(items) - 10} more items"
                        )

                # Validate content before copying
                validation_result = self.validate_download_content(source_dir)
                result["validation_result"] = validation_result

                if not validation_result["valid"]:
                    result["error"] = (
                        f"Content validation failed: {'; '.join(validation_result['errors'])}"
                    )
                    logger.error(f"[Games Hub Download] {result['error']}")
                    return result

                # Create target directory and copy files
                logger.info(
                    f"[Games Hub Download] Creating target directory: {self.hub_root_path}"
                )
                self.hub_root_path.mkdir(parents=True, exist_ok=True)

                # Remove existing content if any
                if self.hub_root_path.exists():
                    for item in self.hub_root_path.iterdir():
                        if item.is_dir():
                            shutil.rmtree(item)
                        else:
                            item.unlink()

                logger.info(
                    f"[Games Hub Download] Copying files from {source_dir} to {self.hub_root_path}"
                )

                # Copy all content
                for item in source_dir.iterdir():
                    target = self.hub_root_path / item.name
                    if item.is_dir():
                        shutil.copytree(item, target, dirs_exist_ok=True)
                    else:
                        shutil.copy2(item, target)

                # Final verification
                if not self.check_installation():
                    result["error"] = (
                        "Installation verification failed after copying files"
                    )
                    logger.error(f"[Games Hub Download] {result['error']}")
                    return result

                # Scan games directory and generate fresh config (if enabled)
                if self.valves.auto_scan_games:
                    logger.info(
                        "[Games Hub Download] Scanning games directory and generating config..."
                    )
                    scan_result = self.scan_games_directory()

                    if scan_result["success"]:
                        logger.info(
                            f"[Games Hub Download] ✅ Games scan complete: {scan_result['games_found']} games found"
                        )
                        result["games_scanned"] = scan_result["games_found"]
                        result["scan_details"] = scan_result
                    else:
                        logger.warning(
                            f"[Games Hub Download] ⚠️ Games scan failed: {scan_result['error']}"
                        )
                        result["scan_error"] = scan_result["error"]
                else:
                    logger.info(
                        "[Games Hub Download] Auto-scan disabled, skipping games directory scan"
                    )
                    result["games_scanned"] = 0

                result["success"] = True
                logger.info(
                    "[Games Hub Download] ✅ Installation completed successfully!"
                )
                return result

        except urllib.error.URLError as e:
            result["error"] = f"Download failed: {str(e)}"
            logger.error(f"[Games Hub Download] URLError: {result['error']}")
            return result
        except Exception as e:
            result["error"] = f"Unexpected error: {str(e)}"
            logger.error(f"[Games Hub Download] Exception: {result['error']}")
            traceback.print_exc()
            return result

    def get_installation_status(self) -> Dict:
        """Get detailed installation status information."""
        status = {
            "installed": False,
            "hub_path": str(self.hub_root_path),
            "index_exists": False,
            "files_found": [],
            "missing_files": [],
            "missing_dirs": [],
            "total_size": 0,
            "games_count": 0,
        }

        try:
            status["index_exists"] = self.hub_index_path.exists()

            if self.hub_root_path.exists():
                # Essential files based on actual Games Hub structure
                essential_files = [
                    "index.html",
                    "style.css",
                    "main.js",
                    "utils.js",
                    "game-manager.js",
                    "event-manager.js",
                    "theme-manager.js",
                    "ui-manager.js",
                    "config-manager.js",  # This is in root directory
                    # Note: config/games_config.json is generated during scan
                ]

                essential_dirs = ["config", "games"]

                # Catalog all files
                for file_path in self.hub_root_path.rglob("*"):
                    if file_path.is_file():
                        relative_path = file_path.relative_to(self.hub_root_path)
                        status["files_found"].append(str(relative_path))
                        status["total_size"] += file_path.stat().st_size

                        # Count games
                        if str(relative_path).startswith("games/"):
                            status["games_count"] += 1

                # Check for missing essential files
                for essential in essential_files:
                    found = any(
                        essential == found_file or found_file.endswith(f"/{essential}")
                        for found_file in status["files_found"]
                    )
                    if not found:
                        status["missing_files"].append(essential)

                # Check for missing essential directories
                for essential_dir in essential_dirs:
                    dir_path = self.hub_root_path / essential_dir
                    if not dir_path.exists() or not dir_path.is_dir():
                        status["missing_dirs"].append(essential_dir)

                status["installed"] = (
                    len(status["missing_files"]) == 0
                    and len(status["missing_dirs"]) == 0
                )

        except Exception as e:
            logger.error(f"[Installation Status] Error: {e}")

        return status

    def scan_games_directory(self) -> Dict:
        """Scan the games directory and generate/update games_config.json."""
        scan_result = {
            "success": False,
            "games_found": 0,
            "games_config": {},
            "error": None,
            "scanned_files": [],
        }

        try:
            games_dir = self.hub_root_path / "games"
            config_dir = self.hub_root_path / "config"
            config_file = config_dir / "games_config.json"

            if not games_dir.exists():
                scan_result["error"] = "Games directory not found"
                logger.error(f"[Games Scanner] Games directory not found: {games_dir}")
                return scan_result

            logger.info(f"[Games Scanner] Scanning games directory: {games_dir}")

            games_config = {
                "version": "1.0",
                "last_scan": time.strftime("%Y-%m-%d %H:%M:%S"),
                "games": {},
            }

            # Scan for game files and directories
            game_count = 0
            scanned_files = []

            for item in games_dir.iterdir():
                if item.is_file():
                    # Handle individual game files (like .html, .js games)
                    if item.suffix.lower() in [".html", ".js", ".htm"]:
                        game_id = item.stem
                        game_name = (
                            item.stem.replace("_", " ").replace("-", " ").title()
                        )

                        games_config["games"][game_id] = {
                            "name": game_name,
                            "type": "file",
                            "file": item.name,  # JavaScript expects this format
                            "path": f"games/{item.name}",  # Full path for reference
                            "enabled": True,
                            "description": f"Game: {game_name}",
                            "category": "uncategorized",
                        }

                        game_count += 1
                        scanned_files.append(str(item.relative_to(self.hub_root_path)))

                elif item.is_dir():
                    # Handle game directories
                    game_id = item.name
                    game_name = item.name.replace("_", " ").replace("-", " ").title()

                    # Look for main game file in directory
                    main_files = []
                    for game_file in item.iterdir():
                        if game_file.is_file():
                            scanned_files.append(
                                str(game_file.relative_to(self.hub_root_path))
                            )
                            if game_file.suffix.lower() in [".html", ".js", ".htm"]:
                                main_files.append(game_file.name)

                    # Determine main file (prefer index.html, then game.html, then first .html/.js)
                    main_file = None
                    if "index.html" in main_files:
                        main_file = "index.html"
                    elif "game.html" in main_files:
                        main_file = "game.html"
                    elif f"{game_id}.html" in main_files:
                        main_file = f"{game_id}.html"
                    elif main_files:
                        main_file = main_files[0]

                    if main_file:
                        games_config["games"][game_id] = {
                            "name": game_name,
                            "type": "directory",
                            "directory": item.name,
                            "main_file": main_file,
                            "file": f"{item.name}/{main_file}",  # JavaScript expects this format
                            "path": f"games/{item.name}/{main_file}",  # Full path for reference
                            "enabled": True,
                            "description": f"Game: {game_name}",
                            "category": "uncategorized",
                        }

                        game_count += 1

            # Ensure config directory exists
            config_dir.mkdir(exist_ok=True)

            # Write the games config file
            with open(config_file, "w", encoding="utf-8") as f:
                json.dump(games_config, f, indent=2, ensure_ascii=False)

            scan_result.update(
                {
                    "success": True,
                    "games_found": game_count,
                    "games_config": games_config,
                    "scanned_files": scanned_files,
                    "config_file": str(config_file),
                }
            )

            logger.info(
                f"[Games Scanner] ✅ Scan complete: {game_count} games found, config updated"
            )
            return scan_result

        except Exception as e:
            scan_result["error"] = f"Scan failed: {str(e)}"
            logger.error(f"[Games Scanner] ❌ Scan failed: {e}")
            traceback.print_exc()
            return scan_result


class Filter:
    class Valves(BaseModel):
        install_on_startup: bool = Field(
            default=True, description="Enable automatic installation on first use"
        )
        force_reinstall: bool = Field(
            default=False, description="Force reinstallation even if hub exists"
        )
        hub_repo_url: str = Field(
            default="https://github.com/pkeffect/games_hub/archive/refs/heads/main.zip",
            description="GitHub repository ZIP URL for Games Hub source",
        )
        trusted_domains: str = Field(
            default="github.com,raw.githubusercontent.com",
            description="Comma-separated list of trusted domains for downloads",
        )
        download_timeout: int = Field(
            default=30, description="Download timeout in seconds"
        )
        show_status_messages: bool = Field(
            default=True, description="Show installation and launch status messages"
        )
        debug_mode: bool = Field(
            default=False, description="Enable detailed debug logging"
        )
        auto_scan_games: bool = Field(
            default=True,
            description="Automatically scan games directory and generate config after installation",
        )
        custom_cache_path: str = Field(
            default="",
            description="Optional: Override automatic cache directory discovery with a custom path (leave empty for auto-detection)",
        )

    def __init__(self):
        self.toggle = True
        self.icon = """data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" d="M14.25 6.087c0-.355.186-.676.401-.959.221-.29.349-.634.349-1.003 0-1.036-1.007-1.875-2.25-1.875s-2.25.84-2.25 1.875c0 .369.128.713.349 1.003.215.283.401.604.401.959v0a.64.64 0 0 1-.657.643 48.39 48.39 0 0 1-4.163-.3c.186 1.613.293 3.25.315 4.907a.656.656 0 0 1-.658.663v0c-.355 0-.676-.186-.959-.401a1.647 1.647 0 0 0-1.003-.349c-1.036 0-1.875 1.007-1.875 2.25s.84 2.25 1.875 2.25c.369 0 .713-.128 1.003-.349.283-.215.604-.401.959-.401v0c.31 0 .555.26.532.57a48.039 48.039 0 0 1-.642 5.056c1.518.19 3.058.309 4.616.354a.64.64 0 0 0 .657-.643v0c0-.355-.186-.676-.401-.959a1.647 1.647 0 0 1-.349-1.003c0-1.035 1.008-1.875 2.25-1.875 1.243 0 2.25.84 2.25 1.875 0 .369-.128.713-.349 1.003-.215.283-.4.604-.4.959v0c0 .333.277.599.61.58a48.1 48.1 0 0 0 5.427-.63 48.05 48.05 0 0 0 .582-4.717.532.532 0 0 0-.533-.57v0c-.355 0-.676.186-.959.401-.29.221-.634.349-1.003.349-1.035 0-1.875-1.007-1.875-2.25s.84-2.25 1.875-2.25c.37 0 .713.128 1.003.349.283.215.604.401.959.401v0a.656.656 0 0 0 .658-.663 48.422 48.422 0 0 0-.37-5.36c-1.886.342-3.81.574-5.766.689a.578.578 0 0 1-.61-.58v0Z" /></svg>"""
        self.valves = self.Valves()
        self.downloader = HubDownloadManager(self.valves)

    async def _initialize_hub_if_needed(self):
        """Initialize Games Hub with robust download and validation."""
        global _HUB_READY, _IS_INITIALIZING

        # Fast path: If already ready and not forcing reinstall
        if _HUB_READY and not self.valves.force_reinstall:
            return {"success": True, "message": "Hub already ready"}

        async with _INIT_LOCK:
            # Re-check after acquiring lock
            if _HUB_READY and not self.valves.force_reinstall:
                return {"success": True, "message": "Hub already ready"}

            # Prevent re-entry
            if _IS_INITIALIZING:
                return {
                    "success": False,
                    "message": "Initialization already in progress",
                }

            _IS_INITIALIZING = True
            logger.info("[Games Hub Init] Starting initialization process...")

            try:
                # Check if auto-installation is disabled
                if not self.valves.install_on_startup:
                    logger.warning("[Games Hub Init] Auto-installation disabled")
                    # Check if files exist anyway
                    _HUB_READY = self.downloader.check_installation()
                    if _HUB_READY:
                        return {
                            "success": True,
                            "message": "Hub found (auto-install disabled)",
                        }
                    else:
                        return {
                            "success": False,
                            "message": "Hub not found and auto-install disabled",
                        }

                # Check current installation status
                status = self.downloader.get_installation_status()
                if self.valves.debug_mode:
                    logger.info(f"[Games Hub Init] Current status: {status}")

                # Decide if we need to download
                need_download = (
                    not status["installed"]
                    or self.valves.force_reinstall
                    or len(status["missing_files"]) > 0
                )

                if not need_download:
                    logger.info(
                        "[Games Hub Init] Installation appears complete, skipping download"
                    )
                    _HUB_READY = True
                    return {"success": True, "message": "Hub already installed"}

                # Perform download and installation
                logger.info("[Games Hub Init] Starting download and installation...")
                download_result = await self.downloader.download_and_extract_hub()

                if download_result["success"]:
                    _HUB_READY = True
                    logger.info(
                        "[Games Hub Init] ✅ Initialization completed successfully"
                    )

                    # Reset force_reinstall flag
                    if self.valves.force_reinstall:
                        self.valves.force_reinstall = False
                        logger.info("[Games Hub Init] Reset force_reinstall flag")

                    return {
                        "success": True,
                        "message": f"Hub installed successfully ({download_result['files_extracted']} files, {download_result.get('games_scanned', 0)} games scanned)",
                        "details": download_result,
                    }
                else:
                    _HUB_READY = False
                    logger.error(
                        f"[Games Hub Init] ❌ Installation failed: {download_result['error']}"
                    )
                    return {
                        "success": False,
                        "message": f"Installation failed: {download_result['error']}",
                        "details": download_result,
                    }

            except Exception as e:
                logger.error(f"[Games Hub Init] CRITICAL ERROR: {e}")
                traceback.print_exc()
                _HUB_READY = False
                return {"success": False, "message": f"Critical error: {e}"}
            finally:
                _IS_INITIALIZING = False
                logger.info(
                    f"[Games Hub Init] Process complete. Hub ready: {_HUB_READY}"
                )

    def _is_games_command(self, content: str) -> bool:
        """Check if content contains the games command."""
        return content and content.lower().strip().startswith("!games")

    def _parse_games_command(self, content: str) -> Dict:
        """Parse games command and return command details."""
        if not content or not content.lower().strip().startswith("!games"):
            return {"command": None}

        # Remove !games prefix and get the rest
        command_part = content.lower().strip()[6:].strip()  # Remove "!games"

        if not command_part:
            return {"command": "launch"}  # Default command
        elif command_part == "scan":
            return {"command": "scan"}
        else:
            return {"command": "launch"}  # Default to launch for any other text

    def _get_relative_hub_url(self) -> str:
        """
        Generate the relative URL for accessing the Games Hub from the web interface.
        This constructs the web-accessible path that corresponds to our cache directory.
        """
        try:
            # The web server typically maps /cache/ to the backend/data/cache/ directory
            # Our hub is at: backend/data/cache/functions/games_hub/
            # So the web URL should be: /cache/functions/games_hub/index.html
            relative_url = "/cache/functions/games_hub/index.html"
            logger.info(f"[Hub URL] Generated relative URL: {relative_url}")
            return relative_url
        except Exception as e:
            logger.error(f"[Hub URL] Error generating URL: {e}")
            # Fallback to the original hardcoded path
            return "/cache/functions/games_hub/index.html"

    async def _emit_status(
        self,
        emitter: Callable,
        message: str,
        status_type: str = "in_progress",
        done: bool = False,
    ):
        """Emit status message if enabled."""
        if not emitter or not self.valves.show_status_messages:
            return

        try:
            await emitter(
                {
                    "type": "status",
                    "data": {
                        "description": message,
                        "status": status_type,
                        "done": done,
                        "hidden": done,
                    },
                }
            )
        except Exception as e:
            logger.error(f"[Status Emit] Error: {e}")

    async def _handle_games_scan_command(
        self, __event_call__, __event_emitter__
    ) -> None:
        """Handle the !games scan command to regenerate games config."""
        global _HUB_READY

        try:
            # Check if hub is installed
            if not _HUB_READY or not self.downloader.check_installation():
                await self._emit_status(
                    __event_emitter__,
                    "❌ Games Hub not installed. Use !games to install first.",
                    "error",
                    done=True,
                )
                return

            # Show scanning status
            await self._emit_status(
                __event_emitter__, "🔍 Scanning games directory...", "in_progress"
            )

            # Perform the scan
            scan_result = self.downloader.scan_games_directory()

            if scan_result["success"]:
                games_count = scan_result["games_found"]
                scanned_files_count = len(scan_result["scanned_files"])

                status_msg = f"✅ Scan complete! Found {games_count} games from {scanned_files_count} files. Config updated."
                await self._emit_status(__event_emitter__, status_msg, "complete")

                # Clear status after delay
                await asyncio.sleep(3)
                await self._emit_status(__event_emitter__, "", "complete", done=True)
            else:
                error_msg = f"❌ Scan failed: {scan_result['error']}"
                await self._emit_status(
                    __event_emitter__, error_msg, "error", done=True
                )

        except Exception as e:
            logger.error(f"[Games Scan Command] Error: {e}")
            traceback.print_exc()
            await self._emit_status(
                __event_emitter__, f"❌ Scan error: {str(e)}", "error", done=True
            )

    async def _handle_games_command(self, __event_call__, __event_emitter__) -> None:
        """Handle the !games command with comprehensive status reporting."""
        global _HUB_READY, _IS_INITIALIZING

        try:
            # Show initial status
            await self._emit_status(
                __event_emitter__, "🎮 Initializing Games Hub...", "in_progress"
            )

            # Wait for initialization if in progress
            if _IS_INITIALIZING:
                await self._emit_status(
                    __event_emitter__,
                    "⏳ Setup in progress, please wait...",
                    "in_progress",
                )
                async with _INIT_LOCK:
                    pass  # Wait for initialization to complete

            # Check final status
            if not _HUB_READY:
                # Get detailed status for error reporting
                status = self.downloader.get_installation_status()
                error_details = []

                if not status["index_exists"]:
                    error_details.append("index.html missing")
                if status["missing_files"]:
                    error_details.append(
                        f"missing files: {', '.join(status['missing_files'][:3])}"
                    )
                if status["missing_dirs"]:
                    error_details.append(
                        f"missing directories: {', '.join(status['missing_dirs'])}"
                    )
                if len(status["missing_files"]) > 3:
                    error_details.append(
                        f"and {len(status['missing_files']) - 3} more files"
                    )

                error_msg = f"❌ Hub not ready. {'; '.join(error_details) if error_details else 'Check server logs for details.'}"
                await self._emit_status(
                    __event_emitter__, error_msg, "error", done=True
                )
                return

            # Hub is ready - launch it
            await self._emit_status(
                __event_emitter__, "✅ Hub ready! Opening Games Hub...", "complete"
            )

            # Use the environment-agnostic URL generation
            hub_url = self._get_relative_hub_url()
            popup_script = f"""
                window.open(
                    '{hub_url}', 
                    'gamesHub_' + Date.now(), 
                    'width=' + Math.min(screen.availWidth, 1200) + ',height=' + Math.min(screen.availHeight, 800) + ',scrollbars=yes,resizable=yes'
                );
            """

            if __event_call__:
                await __event_call__(
                    {"type": "execute", "data": {"code": popup_script}}
                )

            # Clear status after a delay
            await asyncio.sleep(2)
            await self._emit_status(__event_emitter__, "", "complete", done=True)

        except Exception as e:
            logger.error(f"[Games Command] Error: {e}")
            traceback.print_exc()
            await self._emit_status(
                __event_emitter__, f"❌ Error: {str(e)}", "error", done=True
            )

    async def inlet(
        self,
        body: dict,
        __event_emitter__: Callable[[dict], Any],
        __event_call__: Callable[[dict], Any] = None,
        __user__: Optional[dict] = None,
    ) -> dict:
        """Main inlet handler for the Games Hub Filter."""
        if not self.toggle:
            return body

        messages = body.get("messages", [])
        if not messages:
            return body

        # Find last user message
        last_user_message_content = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                last_user_message_content = msg.get("content", "")
                break

        # Handle games command
        if self._is_games_command(last_user_message_content):
            # Parse the specific games command
            command_details = self._parse_games_command(last_user_message_content)

            if command_details["command"] == "scan":
                # Handle scan command
                asyncio.create_task(
                    self._handle_games_scan_command(__event_call__, __event_emitter__)
                )

                # Modify the user message for scan response
                for msg in reversed(messages):
                    if msg.get("role") == "user":
                        msg["content"] = (
                            "Respond with exactly this message: 'Scanning games directory and updating configuration... 🔍'"
                        )
                        break

            else:
                # Handle regular launch command
                # Initialize hub if needed (this is now the primary entry point)
                init_result = await self._initialize_hub_if_needed()

                if self.valves.debug_mode:
                    logger.info(f"[Games Inlet] Initialization result: {init_result}")

                # Launch the hub regardless of initialization result (handler will check status)
                asyncio.create_task(
                    self._handle_games_command(__event_call__, __event_emitter__)
                )

                # Modify the user message to get a nice response
                for msg in reversed(messages):
                    if msg.get("role") == "user":
                        msg["content"] = (
                            "Respond with exactly this message: 'Opening the 🎮Games Hub now. Enjoy your gaming session! If you experience any issues try running `!games scan`'"
                        )
                        break

        return body
