import { BrowserRouter } from "react-router-dom";
import AppRoutes from "./routes";
import { useEffect, useState } from "react";
import "./Styles/LoadingScreen.css";
import { Grid, Typography } from "@mui/material";

const Container = () => {
  const [initialLoad, setInitialLoad] = useState(true);
  useEffect(() => {
    if (initialLoad) {
      setTimeout(() => {
        setInitialLoad(false);
      }, 2000);
    }
  }, [initialLoad]);
  if (initialLoad) {
    return (
      <div style={{ width: "100%", textAlign: "center" }}>
        <Grid
          container
          className="background-image-loading"
          direction="column"
          alignItems="center"
          justifyContent="center"
          spacing={10}
        >
          <Grid item>
            <img src="/pencilLoading.gif" alt="Logo" className="logo" />
          </Grid>
          <Grid item>
            <Typography>Loading...</Typography>
          </Grid>
        </Grid>
      </div>
    );
  }
  return (
    <BrowserRouter>
      <AppRoutes />
    </BrowserRouter>
  );
};

export default Container;
