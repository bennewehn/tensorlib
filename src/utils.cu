#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "utils.h"

// Function to convert an integer array to a string
char* array_to_string(const int* arr, int size) {
    // Estimate the required length for the final string
    // Each integer can be up to 11 characters long (including negative sign), plus commas, plus null terminator
    int max_length = size * 12; // 11 for integer + 1 for comma or null terminator
    char* result = (char*)malloc(max_length);
    if (result == NULL) {
        printf("Memory allocation failed.\n");
        exit(1);
    }
    
    // Initialize the result string
    result[0] = '\0';
    
    // Temporary buffer to hold the string representation of each integer
    char buffer[12];
    
    for (int i = 0; i < size; i++) {
        // Convert integer to string and store in buffer
        snprintf(buffer, sizeof(buffer), "%d", arr[i]);
        
        // Concatenate buffer to result string
        strcat(result, buffer);
        
        // Add a comma if not the last element
        if (i < size - 1) {
            strcat(result, ",");
        }
    }
    
    return result;
}

void printIndent(int indent) {
  for (int i = 0; i < indent; i++) {
    printf("  ");
  }
}