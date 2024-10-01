#!/bin/bash

# Function to check if a file should be excluded based on .gitignore
should_exclude() {
    git -C "$1" check-ignore -q "$2"
}

# Function to check if a file is a binary file, PDF, image, audio, or directory
is_non_project_file() {
    local mime_type
    mime_type=$(file --mime-type -b "$1")
    [[ $mime_type == application/pdf || $mime_type == image/* || $mime_type == inode/directory || $mime_type == application/* || $mime_type == audio/* ]]
}

# Function to list project files
list_project_files() {
    local script_name="$1"
    local output_file="$2"
    local code_repo_path="$3"
    local find_command="find $code_repo_path -type f ! -name \"$script_name\" ! -name \"$output_file\""

    for folder in "${EXCLUDE_FOLDERS[@]}"; do
        find_command+=" ! -path \"$folder/*\""
    done

    for pattern in "${EXCLUDE_FILES[@]}"; do
        find_command+=" ! -name \"$pattern\""
    done

    eval "$find_command" | while read -r file; do
        if should_exclude "$code_repo_path" "$file" || is_non_project_file "$file"; then
            continue
        fi
        echo "$file"
    done
}

# Function to combine files into one
combine_files() {
    local script_name="$1"
    local output_file="$2"
    local code_repo_path="$3"
    list_project_files "$script_name" "$output_file" "$code_repo_path" | while read -r file; do
        {
            echo "### $file ###"
            cat "$file"
            echo
        } >> "$output_file"
    done
}

# Function to display usage
usage() {
    echo "Usage: $0 [-d] [-o output_file] [-e exclude_folder] [-p code_repo_path]"
    echo "  -d  Dry run, only list files"
    echo "  -o  Specify output file name (default: code.txt)"
    echo "  -e  Add folder to exclude (can be used multiple times)"
    echo "  -p  Path to the code repository (default: current directory)"
    exit 1
}

# Main script execution
main() {
    local dry_run=false
    local output_file="code.txt"
    local code_repo_path="."

    # Parse command line options
    while getopts ":do:e:p:" opt; do
        case $opt in
            d)
                dry_run=true
                ;;
            o)
                output_file=$OPTARG
                ;;
            e)
                EXCLUDE_FOLDERS+=("$OPTARG")
                ;;
            p)
                code_repo_path=$OPTARG
                ;;
            \?)
                echo "Invalid option: -$OPTARG" >&2
                usage
                ;;
            :)
                echo "Option -$OPTARG requires an argument." >&2
                usage
                ;;
        esac
    done

    # Expand tilde to full path for code_repo_path
    code_repo_path=$(eval echo "$code_repo_path")

    # Set default exclude folders after code_repo_path is set
    EXCLUDE_FOLDERS=(
        "$code_repo_path/.git"
        "$code_repo_path/examples"
        "$code_repo_path/node_modules"
        "$code_repo_path/dist"
        "$code_repo_path/build"
        "$code_repo_path/.venv"
        "$code_repo_path/__pycache__"
        "$code_repo_path/vendor"
        "$code_repo_path/coverage"
        "$code_repo_path/.cache"
        "$code_repo_path/.yarn"
    )

    # Add typical lock files to exclusions
    EXCLUDE_FILES=(
        "*.log"
        "*.lock"
        "poetry.lock"
        "Pipfile.lock"
        "yarn.lock"
        "package-lock.json"
        "go.sum"
        "*.tsbuildinfo"
    )

    # Interactive mode for code repository path
    if [[ "$code_repo_path" == "." ]]; then
        read -rp "Enter the path to the code repository (default: current directory): " user_code_repo_path
        if [[ -n "$user_code_repo_path" ]]; then
            code_repo_path=$(eval echo "$user_code_repo_path")
            EXCLUDE_FOLDERS=(
                "$code_repo_path/.git"
                "$code_repo_path/examples"
                "$code_repo_path/node_modules"
                "$code_repo_path/dist"
                "$code_repo_path/build"
                "$code_repo_path/.venv"
                "$code_repo_path/__pycache__"
                "$code_repo_path/vendor"
                "$code_repo_path/coverage"
                "$code_repo_path/.cache"
                "$code_repo_path/.yarn"
            )
        fi
    fi

    # Interactive mode for dry run
    if [[ "$dry_run" = false ]]; then
        read -rp "Do you want to perform a dry run? (y/n): " dry_run_input
        if [[ "$dry_run_input" =~ ^[Yy]$ ]]; then
            dry_run=true
        fi
    fi

    # Interactive mode for output file name
    read -rp "Enter the output file name (default: $output_file): " user_output_file
    if [[ -n "$user_output_file" ]]; then
        output_file=$user_output_file
    fi

    # Interactive mode for excluding additional folders
    read -rp "Enter folder patterns to exclude (comma-separated, relative to the repository root): " user_exclude_folders
    if [[ -n "$user_exclude_folders" ]]; then
        IFS=',' read -r -a user_exclude_folders_array <<< "$user_exclude_folders"
        for pattern in "${user_exclude_folders_array[@]}"; do
            EXCLUDE_FOLDERS+=("$code_repo_path/$pattern")
        done
    fi

    # Get the script's own filename
    local script_name
    script_name=$(basename "$0")

    # Dry run or combine files
    if $dry_run; then
        echo "Dry run: Listing project files..."
        list_project_files "$script_name" "$output_file" "$code_repo_path"
    else
        echo "Combining project files into $output_file..."
        > "$output_file"  # Ensure the output file is empty before writing
        combine_files "$script_name" "$output_file" "$code_repo_path"
        echo "Files combined into $output_file"
    fi
}

main "$@"
