# Stage 1: Build
FROM node:18-alpine as build
WORKDIR /app
COPY package*.json ./
RUN npm install
COPY . .
RUN npm run build

# Stage 2: Serve
FROM nginx:alpine
COPY --from=build /app/dist /usr/share/nginx/html
# Copy custom nginx.conf if needed, but default is usually fine for SPA if configured
# For simple SPA support (redirect 404 to index.html):
RUN echo 'server { listen 80; root /usr/share/nginx/html; include /etc/nginx/mime.types; location / { try_files $uri $uri/ /index.html; } }' > /etc/nginx/conf.d/default.conf

EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
