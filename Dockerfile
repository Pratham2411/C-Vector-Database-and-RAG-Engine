FROM alpine:3.18

# Install build dependencies
RUN apk add --no-cache build-base cmake ninja linux-headers

# Set working directory
WORKDIR /app

# Copy project files
COPY . .

# Build the project
RUN cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
RUN cmake --build build

# Expose the API port
EXPOSE 8080

# Command to run the database
CMD ["./build/db"]
