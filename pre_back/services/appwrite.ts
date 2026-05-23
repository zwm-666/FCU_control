import { Account, Client, Databases } from 'appwrite';

const client = new Client()
    .setEndpoint('https://nyc.cloud.appwrite.io/v1')
    .setProject('69b3e33000227d93484b');

const account = new Account(client);
const databases = new Databases(client);

export { client, account, databases };

