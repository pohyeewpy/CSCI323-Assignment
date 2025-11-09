# ReviewLens frontend (Vite dev server)
FROM node:20-alpine

WORKDIR /web

# Install deps first (leverage Docker cache)
COPY package.json package-lock.json* yarn.lock* pnpm-lock.yaml* ./
RUN if [ -f package-lock.json ]; then npm ci;     elif [ -f yarn.lock ]; then yarn install --frozen-lockfile;     elif [ -f pnpm-lock.yaml ]; then npm i -g pnpm && pnpm i --frozen-lockfile;     else npm install; fi

# Now copy the rest
COPY . .

# Vite dev server on 5173, host binding for container
EXPOSE 5173
ENV HOST=0.0.0.0
CMD ["npm","run","dev","--","--host","0.0.0.0"]
