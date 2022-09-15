import FWCore.ParameterSet.Config as cms
from PhysicsTools.NanoAOD.common_cff import Var

def customise(process):
  process.MessageLogger.cerr.FwkReport.reportEvery = 100
  process.finalGenParticles.select = cms.vstring(
    "drop *",
    "keep++ abs(pdgId) == 15 & (pt > 15 ||  isPromptDecayed() )",#  keep full tau decay chain for some taus
    "keep+ abs(pdgId) == 15 ",  #  keep first gen decay product for all tau
    "+keep abs(pdgId) == 11 || abs(pdgId) == 13 || abs(pdgId) == 15", #keep leptons, with at most one mother back in the history
    "drop abs(pdgId)= 2212 && abs(pz) > 1000", #drop LHC protons accidentally added by previous keeps
    "keep abs(pdgId) == 23 || abs(pdgId) == 24 || abs(pdgId) == 25 || abs(pdgId) == 9990012 || abs(pdgId) == 9900012",   # keep VIP particles
  )

  process.boostedTauTable.variables.dxy = Var("leadChargedHadrCand().dxy()", float,
    doc="d_{xy} of lead track with respect to PV, in cm (with sign)", precision=10)
  process.boostedTauTable.variables.dz = Var("leadChargedHadrCand().dz()", float,
    doc="d_{z} of lead track with respect to PV, in cm (with sign)", precision=14)
  return process