# Putting this on a new GitHub account

## Before anything: make it PRIVATE

The competition closes 19 August. A public repository containing the solution and the
write-up can be read by any other team, and organisers commonly treat publishing a
solution before the deadline as grounds for disqualification. **Choose Private when you
create it.** Switch it to public after results are announced — at that point it becomes a
good portfolio piece.

## The easy way: no command line at all

Recommended, because your machine's GitHub CLI is signed in to a different account and
adding a second one is more trouble than it is worth for this.

1. Sign in to the new GitHub account at **github.com**.
2. Click the **+** in the top-right, choose **New repository**.
3. Fill in:
   - **Repository name:** `oos-tag-holder-challenge`
   - **Visibility:** **Private**
   - Leave "Add a README" **unticked** — this bundle already has one.
4. Click **Create repository**.
5. On the empty repository page, click **uploading an existing file**.
6. Unzip this bundle, then drag **everything inside the folder** into the browser window.
   Wait for all files to finish uploading — the big ones take a moment.
7. In the "Commit changes" box, type a short message such as
   `Out-of-state tag holder challenge - full solution` and click **Commit changes**.

That is the whole thing. Nothing to install, nothing to configure.

**Two limits to know:** GitHub's web uploader takes at most 100 files at a time and 25 MB
per file. This bundle is well inside both, but if the browser struggles, upload the
`dist` folder as a second batch.

## The command-line way, if you prefer it

You will need a **Personal Access Token** for the new account, because your machine is
signed in as a different user.

1. On the new account: **Settings > Developer settings > Personal access tokens >
   Tokens (classic) > Generate new token**. Tick the **repo** scope. Copy the token —
   GitHub shows it once.
2. Create the empty private repository as in steps 1-4 above.
3. In Terminal, from inside the unzipped folder:

```bash
git init -b main
git add .
git commit -m "Out-of-state tag holder challenge - full solution"
git remote add origin https://github.com/NEW-ACCOUNT-NAME/oos-tag-holder-challenge.git
git push -u origin main
```

When it asks for a username, use the new account name. When it asks for a password,
**paste the token, not your password.**

## Adding your teammate

Repository page > **Settings** > **Collaborators** > **Add people** > their GitHub
username. They get an email invitation.

## What is deliberately not in here

The organisers' data files. That folder is theirs to distribute, not ours, so the code
looks for it on disk instead. Anyone running this needs their own copy of
`Identify_Out_of_State_Tag_Holders` — see `README.md` for how to point at it.
