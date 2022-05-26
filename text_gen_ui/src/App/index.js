import * as React from 'react';
import Avatar from '@mui/material/Avatar';
import Button from '@mui/material/Button';
import CssBaseline from '@mui/material/CssBaseline';
import TextField from '@mui/material/TextField';
import CircularProgress from '@mui/material/CircularProgress';
import Box from '@mui/material/Box';
import TextSnippetIcon from '@mui/icons-material/TextSnippet';import Typography from '@mui/material/Typography';
import Container from '@mui/material/Container';
import { createTheme, ThemeProvider } from '@mui/material/styles';


const theme = createTheme();

export default function TextGenerator() {
  const handleSubmit = (event) => {
    event.preventDefault();
    setGenTextLoading(1);
    const data = new FormData(event.currentTarget);
    const input_str = data.get('input_string');
    fetch('http://localhost:46304/generate_text', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json;charset=utf-8'
        },
        body: JSON.stringify({
            "input_string": input_str,
        })
    })
    .then(response => response.json())
    .then((data) => {
        setGenText(data); 
        setGenTextLoading(0);
    });
  };

  const [genText, setGenText] = React.useState("");
  const [genTextLoading, setGenTextLoading] = React.useState(0);

  return (
    <ThemeProvider theme={theme}>
      <Container component="main" maxWidth="xs">
        <CssBaseline />
        <Box
          sx={{
            marginTop: 8,
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
          }}
        >
          <Avatar sx={{ m: 1, bgcolor: 'secondary.main' }}>
            <TextSnippetIcon />
          </Avatar>
          <Typography component="h1" variant="h5">
            Generate text
          </Typography>
          <Box component="form" onSubmit={handleSubmit} noValidate sx={{ mt: 1 }}>
            <TextField
              margin="normal"
              required
              fullWidth
              id="input_string"
              label="Text to continue"
              name="input_string"
              autoFocus
            />
            <Button
              type="submit"
              fullWidth
              variant="contained"
              sx={{ mt: 3, mb: 2 }}
            >
              Generate
            </Button>
          </Box>

          <Box>
                {genTextLoading === 1 && <CircularProgress />}
                {genTextLoading === 0 && genText === "" && 
                    <Typography component="h3">
                        Please, enter text and click "Generate"
                    </Typography>}
                {genTextLoading === 0 && genText !== "" && 
                    <Typography component="h3">
                        {genText}
                    </Typography>}
          </Box>
        </Box>
      </Container>
    </ThemeProvider>
  );
}