import React from "react";
import "@/App.css";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import { Toaster } from "sonner";
import Layout from "./components/Layout";
import Overview from "./pages/Overview";
import Queue from "./pages/Queue";
import CaseDetail from "./pages/CaseDetail";
import ImportExport from "./pages/ImportExport";
import AuditLog from "./pages/AuditLog";
import Analytics from "./pages/Analytics";
import ModelPerformance from "./pages/ModelPerformance";
import CategoryQueue from "./pages/CategoryQueue";
import Compare from "./pages/Compare";
import EvidenceSearch from "./pages/EvidenceSearch";

function App() {
  React.useEffect(() => {
    const setTitle = () => { document.title = "Hencheck · DelDOT DMV Casework"; };
    setTitle();
    const id = setInterval(setTitle, 1000);
    return () => clearInterval(id);
  }, []);
  return (
    <div className="App">
      <BrowserRouter>
        <Layout>
          <Routes>
            <Route path="/" element={<Overview />} />
            <Route path="/queue" element={<Queue />} />
            <Route path="/warranted" element={<CategoryQueue classKey="review_warranted" title="Review Warranted" accent="#EF4444" />} />
            <Route path="/not-warranted" element={<CategoryQueue classKey="review_not_warranted" title="Review Not Warranted" accent="#10B981" />} />
            <Route path="/insufficient" element={<CategoryQueue classKey="insufficient_information" title="Insufficient Information" accent="#F59E0B" />} />
            <Route path="/case/:id" element={<CaseDetail />} />
            <Route path="/compare" element={<Compare />} />
            <Route path="/evidence" element={<EvidenceSearch />} />
            <Route path="/analytics" element={<Analytics />} />
            <Route path="/model" element={<ModelPerformance />} />
            <Route path="/audit" element={<AuditLog />} />
            <Route path="/import-export" element={<ImportExport />} />
          </Routes>
        </Layout>
        <Toaster theme="dark" position="bottom-right" richColors closeButton />
      </BrowserRouter>
    </div>
  );
}

export default App;
