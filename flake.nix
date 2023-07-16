{
  nixConfig = {
    extra-substituters = [
      "https://cuda-maintainers.cachix.org"
    ];
    extra-trusted-public-keys = [
      "cuda-maintainers.cachix.org-1:0dq3bujKpuEPMCX6U4WylrUDZ9JyUG0VpVZa7CNfq5E="
    ];
  };

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixpkgs-unstable";
  };

  outputs = {
    self,
    nixpkgs,
  }: let
    system = "x86_64-linux";
    pkgs = import nixpkgs {
      inherit system;
      config.allowUnfree = true;
    };
  in {
    devShells.${system}.default = pkgs.mkShellNoCC {
      packages = with pkgs; [
        # TODO: migrate to 311 when nixpkgs fixes derivations
        python310Packages.transformers
        python310Packages.torch # TODO use cachix for cuda version
        python310Packages.datasets
        python310Packages.soundfile
        python310Packages.librosa
        python310Packages.evaluate
        python310Packages.jiwer
      ];
    };
  };
}
