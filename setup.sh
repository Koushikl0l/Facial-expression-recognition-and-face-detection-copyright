mkdir -p ~/.streamlit/


echo "\
[server]\n\
headless = true\n\
port = $PORT\n\
enableCORS = false\n\
\n\
" > ~/.streamlit/config.toml

echo "\
[general]\n\
email = \"koushikahamed2@gmail.com.br\"\n\
" > ~/.streamlit/credentials.toml
© 2021 GitHub, Inc.
